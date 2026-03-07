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
        self.pe = PositionalEncoding3D(num_freqs=10)
        
        # 回归最本真、最轻量的 79 维结构
        # 将 GELU 替换为 Sine，增强高频纹理生成能力 
        self.inr_mlp = nn.Sequential( 
            nn.Linear(79, 256), Sine(), 
            nn.Linear(256, 256), Sine(), 
            nn.Linear(256, 3) 
        )
        
    def forward(self, lr_seq, coords_xyt):
        B, T, C, H, W = lr_seq.shape
        
        with torch.no_grad():
            lr_seq_input = lr_seq.reshape(B*T, C, H, W) * 2.0 - 1.0
            
            latents = []
            for i in range(B * T):
                single_frame = lr_seq_input[i:i+1].contiguous()
                # 🚀 剥离 autocast 后，这里也是纯 FP32 极速推理
                l = self.encoder.encode(single_frame).latent_dist.mode()
                latents.append(l)
            
            latent = torch.cat(latents, dim=0) * self.encoder.config.scaling_factor
            latent = latent.reshape(B, T, 16, H // 8, W // 8)
            
            f_prev = latent[:, 0].contiguous()
            f_curr = latent[:, 1].contiguous()
            f_next = latent[:, 2].contiguous()
        
        fused_feat = self.t_aiem(f_prev, f_curr, f_next) 
        
        spatial_coords = coords_xyt[..., :2].unsqueeze(1) 
        
        sampled_feat = F.grid_sample(fused_feat, spatial_coords, align_corners=True)
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1).contiguous()
        
        encoded_coords = self.pe(coords_xyt)
        
        inr_input = torch.cat([sampled_feat, encoded_coords], dim=-1).contiguous() 
        pred_rgb_points = self.inr_mlp(inr_input) 
        
        return pred_rgb_points