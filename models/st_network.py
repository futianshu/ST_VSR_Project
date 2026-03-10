import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models import AutoencoderKL
from peft import LoraConfig
from models.t_aiem import T_AIEM

class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class PositionalEncoding3D(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs) * math.pi)

    def forward(self, coords):
        # 巧妙利用广播：[..., 3, 1] * [10] -> [..., 3, 10]
        coords_freq = coords.unsqueeze(-1) * self.freq_bands
        
        # [..., 3, 10] -> [..., 10, 3]
        coords_freq = coords_freq.transpose(-1, -2)
        
        # 瞬间并行计算所有的 sin 和 cos -> [..., 10, 2, 3]
        pe = torch.stack([torch.sin(coords_freq), torch.cos(coords_freq)], dim=-2)
        
        # 一次性展平最后三个维度 (10 * 2 * 3 = 60)，且排序与你的原 for 循环版 100% 绝对等价！
        pe = pe.flatten(start_dim=-3)
        
        return torch.cat([coords, pe], dim=-1) # 返回 63 维

class ST_VSR_Network(nn.Module):
    def __init__(self, sd3_path="stabilityai/stable-diffusion-3-medium-diffusers"):
        super().__init__()
        
        print("🚀 正在初始化 SD3 VAE 作为生成先验提取器...")
        self.encoder = AutoencoderKL.from_pretrained(sd3_path, subfolder="vae")
        
        vae_target_modules = ['encoder.conv_in', 'encoder.down_blocks.0.resnets.0.conv1', 'encoder.down_blocks.0.resnets.0.conv2', 'encoder.down_blocks.0.resnets.1.conv1', 
                              'encoder.down_blocks.0.resnets.1.conv2', 'encoder.down_blocks.0.downsamplers.0.conv', 'encoder.down_blocks.1.resnets.0.conv1',
                              'encoder.down_blocks.1.resnets.0.conv2', 'encoder.down_blocks.1.resnets.0.conv_shortcut', 'encoder.down_blocks.1.resnets.1.conv1', 'encoder.down_blocks.1.resnets.1.conv2', 
                              'encoder.down_blocks.1.downsamplers.0.conv', 'encoder.down_blocks.2.resnets.0.conv1', 'encoder.down_blocks.2.resnets.0.conv2',
                              'encoder.down_blocks.2.resnets.0.conv_shortcut', 'encoder.down_blocks.2.resnets.1.conv1', 'encoder.down_blocks.2.resnets.1.conv2', 'encoder.down_blocks.2.downsamplers.0.conv',
                              'encoder.down_blocks.3.resnets.0.conv1', 'encoder.down_blocks.3.resnets.0.conv2', 'encoder.down_blocks.3.resnets.1.conv1', 'encoder.down_blocks.3.resnets.1.conv2', 
                              'encoder.mid_block.attentions.0.to_q', 'encoder.mid_block.attentions.0.to_k', 'encoder.mid_block.attentions.0.to_v', 'encoder.mid_block.attentions.0.to_out.0', 
                              'encoder.mid_block.resnets.0.conv1', 'encoder.mid_block.resnets.0.conv2', 'encoder.mid_block.resnets.1.conv1', 'encoder.mid_block.resnets.1.conv2', 'encoder.conv_out', 'quant_conv']
        vae_lora_config = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian", target_modules=vae_target_modules)
        self.encoder.add_adapter(vae_lora_config, adapter_name="default")
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        self.t_aiem = T_AIEM(channels=16)
        
        # ========== 【新增：局部上下文特征增强层】 ========== 
        # 提取 3x3 范围的局部结构，并从 16 维升维到 64 维 
        self.feat_proj = nn.Sequential( 
            nn.Conv2d(16, 64, 3, 1, 1), 
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(64, 64, 3, 1, 1) 
        ) 
        
        # ========== 【🌟 核心修复 1：补充像素级空间锚点】 ========== 
        # 弥补 VAE 8倍压缩丢失的物理边缘，直接从 LR 提取高频结构
        self.shallow_cnn = nn.Sequential( 
            nn.Conv2d(3, 64, 3, 1, 1), 
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(64, 64, 3, 1, 1) 
        ) 
        # =========================================================

        self.pe = PositionalEncoding3D(num_freqs=10)
        
        # ========== 【🌟 核心修复 2：解决频率灾难与梯度断崖】 ========== 
        # 维度依然是: 64(融合特征) + 63(PE坐标) = 127
        # 彻底废弃 Sine()，改用 LeakyReLU，完美适配 PositionalEncoding！
        self.inr_mlp = nn.Sequential( 
            nn.Linear(127, 256), 
            nn.LeakyReLU(0.2, True), 
            nn.Linear(256, 256), 
            nn.LeakyReLU(0.2, True), 
            nn.Linear(256, 3) 
        )
        
        # 废弃 1e-5 陷阱，使用科学初始化唤醒梯度
        with torch.no_grad():
            # 仅将最后一层权重适当缩小，既保证初始残差平稳，又能让梯度完美回传！
            self.inr_mlp[4].weight.mul_(0.01)
            self.inr_mlp[4].bias.zero_()
        # =========================================================
        
    def forward(self, lr_seq, coords_xyt, chunk_size=None):
        B, T, C, H, W = lr_seq.shape
        
        with torch.no_grad():
            lr_seq_input = lr_seq.reshape(B*T, C, H, W) * 2.0 - 1.0
            
            # ❌ 必须删掉 with torch.autocast 这行以及它的缩进！ 
            # 🔥 V100 专属：直接裸跑 FP32，既快又 100% 杜绝数值溢出 
            latent = self.encoder.encode(lr_seq_input).latent_dist.mode() 
            
            # 🔥 补充 SD3 特有的 shift_factor 
            shift_factor = getattr(self.encoder.config, "shift_factor", 0.0609) 
            scaling_factor = getattr(self.encoder.config, "scaling_factor", 1.5305) 
            
            # 严格遵循 SD3 官方提取规范：先减去偏移，再乘缩放 
            latent = (latent.float() - shift_factor) * scaling_factor 
            latent = latent.reshape(B, T, 16, H // 8, W // 8)
            
            f_prev = latent[:, 0].contiguous()
            f_curr = latent[:, 1].contiguous()
            f_next = latent[:, 2].contiguous()
        
        fused_feat = self.t_aiem(f_prev, f_curr, f_next) 
        
        # ========== 【新增：应用局部上下文投影】 ========== 
        fused_feat = self.feat_proj(fused_feat) 
        # ================================================ 
        
        # ========== 【新增：提取中心帧作为残差底图】 ========== 
        # lr_seq[:, 1] 是当前时刻的 LR 帧 
        # 提取中心帧
        lr_curr = lr_seq[:, 1].contiguous() 

        # ========== 【🌟 核心修复 3：高低频特征双轨融合】 ==========
        # 1. 提取 LR 图像的浅层特征 (保留了坚硬清晰的梯子、人物轮廓)
        lr_shallow_feat = self.shallow_cnn(lr_curr) 
        # 2. 将微缩的 8x8 VAE 语义特征双线性放大，对齐到 LR 尺寸
        fused_feat_up = F.interpolate(fused_feat, size=(H, W), mode='bilinear', align_corners=False)
        # 3. 强强联手：大模型语义先验(VAE) + 物理边缘结构(LR)
        final_feat = lr_shallow_feat + fused_feat_up
        # =========================================================

        if chunk_size is None or self.training: 
            spatial_coords = coords_xyt[..., :2].unsqueeze(1) 
            
            base_rgb = F.grid_sample(lr_curr, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False) 
            base_rgb = base_rgb.squeeze(2).permute(0, 2, 1).contiguous() 
            
            # 🌟 修复 4：改为对融合后的高清 final_feat 进行采样
            sampled_feat = F.grid_sample(final_feat, spatial_coords, padding_mode='border', align_corners=False) 
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1).contiguous() 
            
            # ========== 【🌟 核心修复 5：绝对坐标转局部相对坐标】 ==========
            device = coords_xyt.device
            # 极其巧妙的数学运算：将全局 [-1, 1] 转化为 LR 像素内部的相对偏移 [-1, 1]
            rel_xy = ((coords_xyt[..., :2] + 1.0) * torch.tensor([W / 2.0, H / 2.0], device=device)) % 1.0 * 2.0 - 1.0
            # 空间用相对坐标，时间 t 保持绝对坐标
            rel_coords = torch.cat([rel_xy, coords_xyt[..., 2:3]], dim=-1)
            encoded_coords = self.pe(rel_coords) 
            # =========================================================
            
            inr_input = torch.cat([sampled_feat, encoded_coords], dim=-1).contiguous() 
            pred_rgb_points = self.inr_mlp(inr_input) 
            
            return pred_rgb_points + base_rgb 
        else: 
            B, N, _ = coords_xyt.shape 
            out_list = [] 
            for i in range(0, N, chunk_size): 
                coords_chunk = coords_xyt[:, i:i+chunk_size, :] 
                spatial_coords = coords_chunk[..., :2].unsqueeze(1) 
                
                base_rgb = F.grid_sample(lr_curr, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False) 
                base_rgb = base_rgb.squeeze(2).permute(0, 2, 1).contiguous() 
                
                # 🌟 改为采样 final_feat
                sampled_feat = F.grid_sample(final_feat, spatial_coords, padding_mode='border', align_corners=False) 
                sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1).contiguous() 
                
                # 🌟 同步转换块内的相对坐标
                device = coords_chunk.device
                rel_xy = ((coords_chunk[..., :2] + 1.0) * torch.tensor([W / 2.0, H / 2.0], device=device)) % 1.0 * 2.0 - 1.0
                rel_coords = torch.cat([rel_xy, coords_chunk[..., 2:3]], dim=-1)
                encoded_coords = self.pe(rel_coords) 
                
                inr_input = torch.cat([sampled_feat, encoded_coords], dim=-1).contiguous() 
                out_chunk = self.inr_mlp(inr_input) 
                
                out_list.append(out_chunk + base_rgb) 
            return torch.cat(out_list, dim=1)