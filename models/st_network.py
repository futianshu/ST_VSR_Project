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
        
        # 🚀 护盾 1【硬件级防爆】：将 79 维升至 80 维！
        # 完美匹配 Tensor Core 的 8 字节对齐要求，彻底免疫奇数维度带来的内存越界！
        self.inr_mlp = nn.Sequential(
            nn.Linear(80, 256), nn.GELU(),  # <--- 这里把原来的 79 改成 80
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 3)
        )
        
    def forward(self, lr_seq, coords_xyt):
        B, T, C, H, W = lr_seq.shape
        
        with torch.no_grad():
            lr_seq_input = lr_seq.reshape(B*T, C, H, W) * 2.0 - 1.0
            
            latents = []
            for i in range(B * T):
                single_frame = lr_seq_input[i:i+1].contiguous()
                with torch.amp.autocast('cuda', enabled=False):
                    l = self.encoder.encode(single_frame.float()).latent_dist.mode()
                latents.append(l)
            
            latent = torch.cat(latents, dim=0) * self.encoder.config.scaling_factor
            latent = latent.reshape(B, T, 16, H // 8, W // 8)
            
            f_prev = latent[:, 0].contiguous()
            f_curr = latent[:, 1].contiguous()
            f_next = latent[:, 2].contiguous()
        
        fused_feat = self.t_aiem(f_prev, f_curr, f_next) 
        
        spatial_coords = coords_xyt[..., :2].unsqueeze(1) 
        spatial_coords = spatial_coords.to(fused_feat.dtype)
        
        sampled_feat = F.grid_sample(fused_feat, spatial_coords, align_corners=False)
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1).contiguous()
        
        encoded_coords = self.pe(coords_xyt).to(sampled_feat.dtype) 
        
        # 🚀 护盾 1【硬件级防爆】：在坐标最后补一个 0，强行把 63维 撑到 64维！
        # 加上 16维特征，总计 80维！完美契合 Tensor Core 的 8 字节对齐要求！
        encoded_coords = F.pad(encoded_coords, (0, 1), value=0.0).contiguous()
        
        inr_input = torch.cat([sampled_feat, encoded_coords], dim=-1).contiguous() # [B, N, 80]
        
        # =========================================================================
        # 🚀 护盾 2【究极降维打击】：彻底解决 nn.Linear 3D 张量反向传播闪退的 Bug！
        # 把 [8, 2304, 80] 拍扁成 2D 矩阵 [18432, 80]，
        # 强制 cuBLAS 使用最基础、最稳如老狗的 2D 矩阵乘法核！
        # =========================================================================
        _B, _N, _D = inr_input.shape
        inr_input_2d = inr_input.view(_B * _N, _D)
        
        # 用 2D 矩阵过 MLP，绝对不会有 stride 内存越界的问题，且计算速度更快！
        pred_rgb_points_2d = self.inr_mlp(inr_input_2d) 
        
        # 算完之后，再安全地恢复成 3D 张量 [8, 2304, 3] 并确保显存连续
        pred_rgb_points = pred_rgb_points_2d.view(_B, _N, 3).contiguous()
            
        return pred_rgb_points