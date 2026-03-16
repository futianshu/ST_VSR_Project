import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models import AutoencoderKL
from peft import LoraConfig

class PositionalEncoding3D(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs) * math.pi)

    def forward(self, coords):
        coords_freq = coords.unsqueeze(-1) * self.freq_bands
        coords_freq = coords_freq.transpose(-1, -2)
        pe = torch.stack([torch.sin(coords_freq), torch.cos(coords_freq)], dim=-2)
        pe = pe.flatten(start_dim=-3)
        return torch.cat([coords, pe], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class ST_VSR_Network(nn.Module):
    def __init__(self, sd3_path="stabilityai/stable-diffusion-3-medium-diffusers", use_time_cond=True, use_shallow_cnn=True):
        super().__init__()
        self.use_time_cond = use_time_cond
        self.use_shallow_cnn = use_shallow_cnn
        
        print("🚀 正在初始化 SD3 VAE 作为生成先验提取器...")
        self.encoder = AutoencoderKL.from_pretrained(sd3_path, subfolder="vae")
        
        # [请保留你原来的 vae_target_modules 完整列表，这里为排版简写]
        vae_target_modules = ['encoder.conv_in', 'encoder.down_blocks.0.resnets.0.conv1'] 
        vae_lora_config = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian", target_modules=vae_target_modules)
        self.encoder.add_adapter(vae_lora_config, adapter_name="default")
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # ========== 核心 1：时间条件生成的 VAE 语义解压缩 ==========
        # 如果启用时序融合，输入为 [z_prev, z_curr, z_next, t_map] 共 16*3+1=49 通道
        latent_in_channels = 49 if self.use_time_cond else 16
        self.vae_decoder = nn.Sequential(
            nn.Conv2d(latent_in_channels, 256, 3, 1, 1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        
        # ========== 核心 2：时间条件驱动的物理对齐网络 ==========
        if self.use_shallow_cnn:
            self.shallow_feat_extract = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                ResBlock(64)
            )
            # 如果不启用时序，这一层不会被使用；若启用，输入为 [f_prev, f_curr, f_next, t_map] 共 64*3+1=193 通道
            self.time_cond_fusion = nn.Sequential(
                nn.Conv2d(193, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                ResBlock(64), ResBlock(64),
                nn.Conv2d(64, 64, 3, 1, 1)
            )

        self.pe = PositionalEncoding3D(num_freqs=10)
        
        # ========== 核心 3：专属细化 MLP ==========
        # 输入：64(融合特征) + 63(位置编码) = 127
        self.inr_mlp = nn.Sequential( 
            nn.Linear(127, 256), nn.LeakyReLU(0.2, True), 
            nn.Linear(256, 256), nn.LeakyReLU(0.2, True), 
            nn.Linear(256, 256), nn.LeakyReLU(0.2, True), 
            nn.Linear(256, 3) 
        )
        
        with torch.no_grad():
            self.inr_mlp[-1].weight.mul_(0.01)
            self.inr_mlp[-1].bias.zero_()

    def forward(self, lr_seq, coords_xyt, chunk_size=None):
        B, T, C, H, W = lr_seq.shape
        lr_prev, lr_curr, lr_next = lr_seq[:, 0], lr_seq[:, 1], lr_seq[:, 2]
        
        # 🧠 灵魂设计：将当前查询的时间 t_q 广播成特征图
        # 让具有空间感受野的 CNN 直接看见“目标时间”，从而主动扭曲/移动特征
        t_q = coords_xyt[:, 0, 2].view(B, 1, 1, 1) 
        t_map = t_q.expand(B, 1, H, W).contiguous()
        t_map_latent = t_q.expand(B, 1, H // 8, W // 8).contiguous()

        # --- 1. 物理时空流 (Pixel Space) ---
        if self.use_shallow_cnn:
            if self.use_time_cond:
                f_prev = self.shallow_feat_extract(lr_prev)
                f_curr = self.shallow_feat_extract(lr_curr)
                f_next = self.shallow_feat_extract(lr_next)
                fused_spatial = self.time_cond_fusion(torch.cat([f_prev, f_curr, f_next, t_map], dim=1))
            else:
                fused_spatial = self.shallow_feat_extract(lr_curr) # Ablation: 退化为单帧
        else:
            fused_spatial = 0.0

        # --- 2. 语义生成流 (SD3 VAE) ---
        with torch.no_grad():
            lr_seq_input = lr_seq.reshape(B*T, C, H, W) * 2.0 - 1.0
            latent = self.encoder.encode(lr_seq_input).latent_dist.mode()
            latent = (latent.float() - 0.0609) * 1.5305
            latent = latent.reshape(B, T, 16, H // 8, W // 8)
            z_prev, z_curr, z_next = latent[:, 0], latent[:, 1], latent[:, 2]

        if self.use_time_cond:
            fused_latent = self.vae_decoder(torch.cat([z_prev, z_curr, z_next, t_map_latent], dim=1))
        else:
            fused_latent = self.vae_decoder(z_curr) # Ablation: 退化为单帧

        final_feat = fused_spatial + fused_latent

        # --- 3. INR 连续推理块分发 ---
        if chunk_size is None or self.training:
            return self._forward_chunk(lr_prev, lr_curr, lr_next, final_feat, coords_xyt, H, W)
        else:
            B_c, N, _ = coords_xyt.shape
            out_list = []
            for i in range(0, N, chunk_size):
                out_chunk = self._forward_chunk(lr_prev, lr_curr, lr_next, final_feat, coords_xyt[:, i:i+chunk_size, :], H, W)
                out_list.append(out_chunk)
            return torch.cat(out_list, dim=1)

    def _forward_chunk(self, lr_prev, lr_curr, lr_next, final_feat, coords, H, W):
        spatial_coords = coords[..., :2].unsqueeze(1) 
        
        # 采样生成好的“目标时间特征”
        sampled_feat = F.grid_sample(final_feat, spatial_coords, padding_mode='border', align_corners=False) 
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1).contiguous() 
        
        device = coords.device
        rel_xy = ((coords[..., :2] + 1.0) * torch.tensor([W / 2.0, H / 2.0], device=device)) % 1.0 * 2.0 - 1.0
        rel_coords = torch.cat([rel_xy, coords[..., 2:3]], dim=-1)
        encoded_coords = self.pe(rel_coords) 
        
        inr_input = torch.cat([sampled_feat, encoded_coords], dim=-1).contiguous() 
        pred_residual = self.inr_mlp(inr_input) 

        # ========== 🧠 拯救残差的物理底图混合器 (Dynamic Base Blending) ==========
        if self.use_time_cond:
            t_chunk = coords[..., 2:3] # [B, N, 1]
            w_p = torch.relu(-t_chunk)         # 距离上一帧的权重 (t=-0.5时为0.5)
            w_n = torch.relu(t_chunk)          # 距离下一帧的权重 (t=0.5时为0.5)
            w_c = 1.0 - torch.abs(t_chunk)     # 距离当前帧的权重 (t=0.0时为1.0)
            
            base_p = F.grid_sample(lr_prev, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            base_c = F.grid_sample(lr_curr, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            base_n = F.grid_sample(lr_next, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            
            base_rgb = w_p * base_p + w_c * base_c + w_n * base_n
        else:
            base_rgb = F.grid_sample(lr_curr, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)

        return pred_residual + base_rgb