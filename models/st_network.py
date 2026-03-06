import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models import AutoencoderKL
from peft import LoraConfig
from models.t_aiem import T_AIEM

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
    # 注意：sd3_path 换成你当时跑 DPAS-SR 用的 SD3 预训练模型路径
    def __init__(self, sd3_path="stabilityai/stable-diffusion-3-medium-diffusers"):
        super().__init__()
        
        # 1. 直接挂载真正的 SD3 VAE 骨架
        print("🚀 正在初始化 SD3 VAE 作为生成先验提取器...")
        self.encoder = AutoencoderKL.from_pretrained(sd3_path, subfolder="vae")
        
        # 完美复刻你 DPAS-SR train.py 里的 VAE LoRA 配置
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
        
        # 冻结 VAE，保住你的 32G 显存！
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # 2. T-AIEM 时序对齐融合 (处理 16 通道)
        self.t_aiem = T_AIEM(channels=16)
        
        # 3. 三维时空隐式解码器 3D-INR (16特征 + 63坐标 = 79)
        self.pe = PositionalEncoding3D(num_freqs=10)
        self.inr_mlp = nn.Sequential(
            nn.Linear(79, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 3)
        )
        
    def forward(self, lr_seq, coords_xyt):
        B, T, C, H, W = lr_seq.shape
        
        with torch.no_grad():
            # SD3 VAE 的输入习惯是 [-1, 1]，而我们普通图片加载是 [0, 1]
            lr_seq_input = lr_seq.view(B*T, C, H, W) * 2.0 - 1.0
            
            # 提取你在 DPAS-SR 里的核心结晶：潜空间特征！(为了稳定提取特征，这里使用 .mode() 而不是 sample)
            latent = self.encoder.encode(lr_seq_input).latent_dist.mode() * self.encoder.config.scaling_factor
            
            # 恢复形状: [B, 3, 16, H/8, W/8]
            latent = latent.view(B, T, 16, H // 8, W // 8)
            f_prev, f_curr, f_next = latent[:, 0], latent[:, 1], latent[:, 2]
        
        # 潜空间时序对齐 (T-AIEM) —— 由于尺寸只有 1/8，运算极度轻量！
        fused_feat = self.t_aiem(f_prev, f_curr, f_next) 
        
        # 隐式渲染 (注意：特征图虽然被 VAE 缩小了 8 倍，但 grid_sample 接受的是归一化坐标 [-1,1]，完全不受绝对尺寸影响，隐式网络的魅力就在于此！)
        spatial_coords = coords_xyt[..., :2].unsqueeze(1) 
        sampled_feat = F.grid_sample(fused_feat, spatial_coords, align_corners=False)
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1) # [B, N, 16]
        
        encoded_coords = self.pe(coords_xyt) # [B, N, 63]
        
        inr_input = torch.cat([sampled_feat, encoded_coords], dim=-1) # [B, N, 79]
        pred_rgb_points = self.inr_mlp(inr_input) 
        
        return pred_rgb_points