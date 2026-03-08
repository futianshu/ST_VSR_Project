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
        pe = []
        for freq in self.freq_bands:
            pe.append(torch.sin(coords * freq))
            pe.append(torch.cos(coords * freq))
        return torch.cat([coords] + pe, dim=-1) # 63维

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
        # ================================================= 
        
        self.pe = PositionalEncoding3D(num_freqs=10)
        
        # 维度调整：64(特征) + 63(坐标) = 127 
        self.inr_mlp = nn.Sequential( 
            nn.Linear(127, 256), Sine(), 
            nn.Linear(256, 256), Sine(), 
            nn.Linear(256, 3) 
        )
        
        # ========== 【新增：SIREN 专属数学初始化】 ========== 
        # 第一层：确保输入坐标频率的均匀分布 
        with torch.no_grad(): 
            self.inr_mlp[0].weight.uniform_(-1 / 127, 1 / 127)  # 改为 127 
            c = math.sqrt(6 / 256) / 30.0 
            self.inr_mlp[2].weight.uniform_(-c, c) 
            
            # 🔥 预测残差的层，初始化为极小值，让初始输出几乎为 0，收敛会极其顺滑 
            self.inr_mlp[4].weight.uniform_(-1e-5, 1e-5) 
            self.inr_mlp[4].bias.zero_()
        # ===================================================
        
    def forward(self, lr_seq, coords_xyt, chunk_size=None):
        B, T, C, H, W = lr_seq.shape
        
        with torch.no_grad():
            lr_seq_input = lr_seq.reshape(B*T, C, H, W) * 2.0 - 1.0
            
            # 🔥 取消 for 循环！开启 bfloat16 混合精度并行提取！ 
            with torch.autocast('cuda', dtype=torch.bfloat16): 
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
        lr_curr = lr_seq[:, 1].contiguous() 
        # =================================================== 

        # ========== 【新增：分块推理机制】 ========== 
        if chunk_size is None or self.training: 
            # 训练时一次性计算 
            spatial_coords = coords_xyt[..., :2].unsqueeze(1) 
            
            # --- 新增：采样底图 --- 
            base_rgb = F.grid_sample(lr_curr, spatial_coords, mode='bicubic', padding_mode='border', align_corners=True) 
            base_rgb = base_rgb.squeeze(2).permute(0, 2, 1).contiguous() 
            
            sampled_feat = F.grid_sample(fused_feat, spatial_coords, padding_mode='border', align_corners=True) 
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1).contiguous() 
            encoded_coords = self.pe(coords_xyt) 
            inr_input = torch.cat([sampled_feat, encoded_coords], dim=-1).contiguous() 
            
            # MLP 此时会自动学到预测“残差” 
            pred_rgb_points = self.inr_mlp(inr_input) 
            
            # --- 返回 残差 + 底图 --- 
            return pred_rgb_points + base_rgb 
        else: 
            # 推理评估时分块计算，彻底杜绝 OOM 
            B, N, _ = coords_xyt.shape 
            out_list = [] 
            for i in range(0, N, chunk_size): 
                coords_chunk = coords_xyt[:, i:i+chunk_size, :] 
                spatial_coords = coords_chunk[..., :2].unsqueeze(1) 
                
                # --- 新增：分块采样底图 --- 
                base_rgb = F.grid_sample(lr_curr, spatial_coords, mode='bicubic', padding_mode='border', align_corners=True) 
                base_rgb = base_rgb.squeeze(2).permute(0, 2, 1).contiguous() 
                
                sampled_feat = F.grid_sample(fused_feat, spatial_coords, padding_mode='border', align_corners=True) 
                sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1).contiguous() 
                encoded_coords = self.pe(coords_chunk) 
                inr_input = torch.cat([sampled_feat, encoded_coords], dim=-1).contiguous() 
                out_chunk = self.inr_mlp(inr_input) 
                
                # --- 分块残差相加 --- 
                out_list.append(out_chunk + base_rgb) 
            return torch.cat(out_list, dim=1) 
        # ============================================