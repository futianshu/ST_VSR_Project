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
        
        # ========== 🚀 终极防崩溃：动态计算 8 的倍数 Padding ==========
        # 训练时 patch_size 是 8 的倍数，这里不会触发；推理时遇到任意分辨率，自动补齐
        H_pad = math.ceil(H / 8) * 8
        W_pad = math.ceil(W / 8) * 8
        pad_h = H_pad - H
        pad_w = W_pad - W
        
        # 🧠 灵魂设计：将当前查询的时间 t_q 广播成特征图
        t_q = coords_xyt[:, 0, 2].view(B, 1, 1, 1) 
        t_map = t_q.expand(B, 1, H, W).contiguous()
        # 注意：这里的 Latent 时间图必须和 Pad 后的尺寸对齐
        t_map_latent = t_q.expand(B, 1, H_pad // 8, W_pad // 8).contiguous()

        # --- 1. 物理时空流 (Pixel Space) ---
        # 物理流不经过 VAE 压缩，继续保留极其精确的原始 H, W 尺寸
        if self.use_shallow_cnn:
            if self.use_time_cond:
                f_prev = self.shallow_feat_extract(lr_prev)
                f_curr = self.shallow_feat_extract(lr_curr)
                f_next = self.shallow_feat_extract(lr_next)
                # 强制 t_map 对齐 f_prev 的精度
                fused_spatial = self.time_cond_fusion(torch.cat([f_prev, f_curr, f_next, t_map.to(f_prev.dtype)], dim=1))
            else:
                fused_spatial = self.shallow_feat_extract(lr_curr) 
        else:
            fused_spatial = 0.0

        # --- 2. 语义生成流 (SD3 VAE) ---
        with torch.no_grad():
            # 强制关闭 autocast，让 VAE 运行在纯 FP32 下，避开 V100 硬件 Bug
            with torch.cuda.amp.autocast(enabled=False):
                lr_seq_input = lr_seq.float().reshape(B*T, C, H, W) * 2.0 - 1.0
                
                # 🚀 执行边缘 Pad，安全补齐至 8 的倍数
                if pad_h > 0 or pad_w > 0:
                    lr_seq_input = F.pad(lr_seq_input, (0, pad_w, 0, pad_h), mode='replicate')
                
                vae_chunk_size = 12 
                latents = []
                for i in range(0, B*T, vae_chunk_size):
                    chunk_in = lr_seq_input[i:i+vae_chunk_size].contiguous()
                    chunk_latent = self.encoder.encode(chunk_in).latent_dist.mode()
                    latents.append(chunk_latent)
                latent = torch.cat(latents, dim=0)
            
            latent = (latent.float() - 0.0609) * 1.5305
            latent = latent.reshape(B, T, 16, H_pad // 8, W_pad // 8)
            z_prev, z_curr, z_next = latent[:, 0], latent[:, 1], latent[:, 2]

        if self.use_time_cond:
            fused_latent = self.vae_decoder(torch.cat([z_prev, z_curr, z_next, t_map_latent.to(z_prev.dtype)], dim=1))
        else:
            fused_latent = self.vae_decoder(z_curr) 
            
        # 🚀 还原裁剪：将 VAE 生成的特征图裁剪回原始长宽，以便与物理流 100% 对齐
        if pad_h > 0 or pad_w > 0:
            fused_latent = fused_latent[:, :, :H, :W]

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
        
        # =========================================================================
        # 🚀 终极排雷：关闭 autocast，强制转为 FP32。
        # 彻底避开 V100 在反向传播时处理 200 万行超大矩阵的 FP16 溢出 Bug！
        with torch.cuda.amp.autocast(enabled=False):
            inr_input = torch.cat([sampled_feat.float(), encoded_coords.float()], dim=-1).contiguous()
            pred_residual = self.inr_mlp(inr_input) 
        # =========================================================================

        # ========== 🧠 拯救残差的物理底图混合器 (Dynamic Base Blending) ==========
        if self.use_time_cond:
            t_chunk = coords[..., 2:3] # [B, N, 1]
            w_p = torch.relu(-t_chunk)         # 距离上一帧的权重 (t=-0.5时为0.5)
            w_n = torch.relu(t_chunk)          # 距离下一帧的权重 (t=0.5时为0.5)
            w_c = 1.0 - torch.abs(t_chunk)     # 距离当前帧的权重 (t=0.0时为1.0)
            
            base_p = F.grid_sample(lr_prev, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            base_c = F.grid_sample(lr_curr, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            base_n = F.grid_sample(lr_next, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            
            base_rgb = w_p.to(base_p.dtype) * base_p + w_c.to(base_c.dtype) * base_c + w_n.to(base_n.dtype) * base_n
        else:
            base_rgb = F.grid_sample(lr_curr, spatial_coords, mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)

        return pred_residual + base_rgb