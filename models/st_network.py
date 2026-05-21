import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models import AutoencoderKL
from peft import LoraConfig
from torch.utils.checkpoint import checkpoint

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
    # ========== 🌟 终极防核爆修复：引入残差缩放系数 ==========
    def __init__(self, channels, res_scale=0.1): 
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        
    def forward(self, x):
        # 乘以 0.1 后再相加，彻底斩断深层特征累加冲破 FP16 极限的可能！
        return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale

class TSM_ResBlock(nn.Module):
    # ========== 🌟 核心创新：零参数时空感受野膨胀 ==========
    def __init__(self, channels, res_scale=0.1, n_segment=3, fold_div=4): 
        super().__init__()
        self.res_scale = res_scale
        self.n_segment = n_segment # 我们是相邻 3 帧 (prev, curr, next)
        self.fold_div = fold_div   # 默认将 1/4 通道前移，1/4 后移
        
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        
    def forward(self, x):
        # x shape: [B*3, C, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x_view = x.view(n_batch, self.n_segment, c, h, w)
        
        fold = c // self.fold_div
        out = torch.zeros_like(x_view)
        
        # 🚀 TSM 时空魔法：在不增加任何卷积计算的情况下，实现物理特征的跨帧交互
        # 1. 前 1/4 的通道：向左平移 (当前帧借用下一帧的特征，预测未来趋势)
        out[:, :-1, :fold] = x_view[:, 1:, :fold]
        # 2. 接着 1/4 的通道：向右平移 (当前帧借用上一帧的特征，保留历史残影)
        out[:, 1:, fold: 2 * fold] = x_view[:, :-1, fold: 2 * fold]
        # 3. 剩下 2/4 的通道：保持原地不动 (锚定当前帧的绝对物理坐标)
        out[:, :, 2 * fold:] = x_view[:, :, 2 * fold:]
        
        # 重新展平回 [B*3, C, H, W] 送入卷积提取特征
        out = out.view(nt, c, h, w)
        return x + self.conv2(self.relu(self.conv1(out))) * self.res_scale

class ECA_Block(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        # 根据通道数自适应计算 1D 卷积的核大小，避免手动调参
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 用 1D 卷积替代 SE-Net 中沉重的全连接层，参数量极小，不增加显存负担
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, H, W]
        
        # ========== 🌟 核心防核爆修复 ==========
        # 强制将 x 转为 float32 进行池化，防止 H*W(65536) 求和时冲破 FP16 的 65504 极限！
        # 池化完后再转回它原本的精度 (x.dtype)
        y = self.avg_pool(x.float()).to(x.dtype)
        # =======================================
        
        # 维度转换配合 1D 卷积: [B, 1, C] -> Conv1d -> [B, 1, C]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # 乘以 Sigmoid 激活后的通道注意力权重
        return x * self.sigmoid(y)

class SFT_Layer(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        # 用语义特征预测缩放 (Scale) 和平移 (Shift)
        self.scale_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), 
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.shift_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), 
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
        # 零初始化：保证训练初期 SFT 层退化为恒等映射，避免 Loss 爆炸
        with torch.no_grad():
            self.scale_conv[-1].weight.zero_()
            self.scale_conv[-1].bias.zero_()
            self.shift_conv[-1].weight.zero_()
            self.shift_conv[-1].bias.zero_()

    def forward(self, f_physical, f_semantic):
        scale = self.scale_conv(f_semantic)
        shift = self.shift_conv(f_semantic)
        
        # 🌟 防核爆修复 3：用 tanh 把缩放系数温柔地软截断在 [-1, 1] 之间，绝对防止乘法溢出
        scale = torch.tanh(scale)
        
        # SFT 核心公式：自适应仿射变换 + 残差保留
        return f_physical * (scale + 1.0) + shift + f_semantic

class LatentAlign_Block(nn.Module):
    # ========== 🌟 核心创新：潜空间隐式特征对齐 ==========
    def __init__(self, in_channels=16):
        super().__init__()
        # 输入是当前帧和相邻帧的拼接 (16+16=32通道)
        # 输出是 2 个通道的偏移场 (Delta X, Delta Y)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 2, 3, 1, 1)
        )
        
        # 🚀 极致防雷：强行让最后一层初始化为 0。
        # 这样在训练初期，偏移量全为 0，网络等价于没加这个模块（恒等映射），绝对不影响收敛！
        with torch.no_grad():
            self.offset_conv[-1].weight.zero_()
            self.offset_conv[-1].bias.zero_()

    def forward(self, z_curr, z_neighbor):
        # z_curr, z_neighbor shape: [B, C, h, w] (此时分辨率已是 H/8, W/8)
        B, C, h, w = z_curr.shape
        
        # 1. 预测特征级的运动偏移
        feat_cat = torch.cat([z_curr, z_neighbor], dim=1)
        offset = self.offset_conv(feat_cat) # [B, 2, h, w]
        
        # 2. 生成基础网格坐标
        y_coords = (torch.arange(h, dtype=torch.float32, device=z_curr.device) + 0.5) / h * 2.0 - 1.0
        x_coords = (torch.arange(w, dtype=torch.float32, device=z_curr.device) + 0.5) / w * 2.0 - 1.0
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1) # [B, h, w, 2]
        
        # 3. 加上偏移量 (使用 tanh 进行极其温柔的软截断，防止特征被过度拉扯撕裂)
        offset = offset.permute(0, 2, 3, 1) # [B, h, w, 2]
        offset = torch.tanh(offset) * 0.5   # 限制最大偏移不超过图像的一半范围

        # ========== 🌟 终极修复 1：Latent 空间分辨率解耦 ==========
        # 训练时 latent 的大小永远是 8x8 (64/8)。将其换算到当前推理分辨率。
        scale_x = 8.0 / w
        scale_y = 8.0 / h
        scale_tensor = torch.tensor([scale_x, scale_y], device=offset.device, dtype=offset.dtype).view(1, 1, 1, 2)
        offset = offset * scale_tensor
        # ========================================================

        grid = base_grid + offset
        
        # 4. 采样对齐 (🌟 终极防核爆：严格在 FP32 空间执行网格采样)
        with torch.amp.autocast('cuda', enabled=False):
            aligned_neighbor = F.grid_sample(
                z_neighbor.float(), 
                grid.float(), 
                mode='bilinear', 
                padding_mode='border', 
                align_corners=False
            )
            
        return aligned_neighbor.to(z_curr.dtype)

class ST_VSR_Network(nn.Module):
    def __init__(self, sd3_path="stabilityai/stable-diffusion-3-medium-diffusers", use_time_cond=True, use_shallow_cnn=True, use_semantic_prior=True):
        super().__init__()
        self.use_time_cond = use_time_cond
        self.use_shallow_cnn = use_shallow_cnn
        self.use_semantic_prior = use_semantic_prior
        
        print("🚀 正在初始化 SD3 VAE 作为生成先验提取器...")
        self.encoder = AutoencoderKL.from_pretrained(sd3_path, subfolder="vae", local_files_only=True)
        
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

        # ========== 🌟 新增：潜空间对齐模块 ==========
        if self.use_time_cond:
            # 潜特征是 16 通道
            self.latent_align = LatentAlign_Block(in_channels=16)
        # ============================================

        # ========== 🌟 新增：特征域预清洗模块 (Pre-Cleaner) ==========
        # 目的：在送入 VAE 之前，洗掉 JPEG 块效应和高斯噪声，防止 VAE 提取“脏先验”
        self.pre_cleaner = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

        # 🔥 关键初始化：强制最后一层权重和偏置为 0
        # 保证网络在初始阶段执行严格的 Identity Mapping (恒等映射)
        # 这样不会在一开始就破坏输入，确保训练稳健起步
        with torch.no_grad():
            self.pre_cleaner[-1].weight.zero_()
            self.pre_cleaner[-1].bias.zero_()
        # ========================================================
        
        # ========== 核心 2：时间条件驱动的物理对齐网络 ==========
        if self.use_shallow_cnn:
            extract_layers = [nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.2, True)]
            for _ in range(15):  
                # 🌟 终极修复：消融实验（无时序）必须退化为普通 ResBlock，严禁使用 TSM！
                if self.use_time_cond:
                    extract_layers.append(TSM_ResBlock(64))
                else:
                    extract_layers.append(ResBlock(64))
            self.shallow_feat_extract = nn.Sequential(*extract_layers)
            
            # 🚀 提升融合流深度：从 2 个块增加到 10 个块
            fusion_layers = [nn.Conv2d(193, 64, 3, 1, 1), nn.LeakyReLU(0.2, True)]
            for _ in range(10):
                fusion_layers.append(ResBlock(64))
            fusion_layers.append(nn.Conv2d(64, 64, 3, 1, 1))
            self.time_cond_fusion = nn.Sequential(*fusion_layers)

            # ========== 新增：193 通道的 ECA 时空特征筛选器 ==========
            self.eca_layer = ECA_Block(channels=193)
            # =======================================================

        self.pe = PositionalEncoding3D(num_freqs=10)
        

        # ========== 新增：轻量级隐式坐标偏移预测器 (Deformable Offset) ==========
        self.offset_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 2, 3, 1, 1) # 输出 2 通道，对应 (Delta X, Delta Y)
        )

        # ========== 新增：SFT 语义引导空间特征变换模块 ==========
        self.sft_layer = SFT_Layer(channels=64)
        # =======================================================
        
        # 🔥 关键初始化：强制最后一层权重和偏置为 0
        # 保证训练初期没有任何坐标偏移，退化为原始网络，确保稳健收敛
        with torch.no_grad():
            self.offset_conv[-1].weight.zero_()
            self.offset_conv[-1].bias.zero_()
        # ====================================================================


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
        
        # ========== 🌟 终极修复：动态保留原始高频微观纹理 (Dynamic Residual Leakage) ==========
        lr_seq_input = lr_seq.float().reshape(B*T, C, H, W)

        # 提取清洗残差
        clean_residual = self.pre_cleaner(lr_seq_input)

        # 强行打半折 (0.5)：保留 50% 清洗能力去脏斑，释放 50% 原始高频细节去激活 VAE 潜能！
        clean_lr_seq = torch.clamp(lr_seq_input + 0.5 * clean_residual, 0.0, 1.0)

        clean_lr_seq = clean_lr_seq.reshape(B, T, C, H, W)
        # =================================================================================
        
        lr_prev = clean_lr_seq[:, 0].contiguous()
        lr_curr = clean_lr_seq[:, 1].contiguous()
        lr_next = clean_lr_seq[:, 2].contiguous()
        # ========================================================
        
        if torch.is_autocast_enabled():
            lr_prev = lr_prev.half()
            lr_curr = lr_curr.half()
            lr_next = lr_next.half()


        # 动态计算 8 的倍数 Padding
        H_pad = math.ceil(H / 8) * 8
        W_pad = math.ceil(W / 8) * 8
        pad_h = H_pad - H
        pad_w = W_pad - W
        
        t_q = coords_xyt[:, 0, 2].view(B, 1, 1, 1) 
        t_map = t_q.expand(B, 1, H, W).contiguous()
        t_map_latent = t_q.expand(B, 1, H_pad // 8, W_pad // 8).contiguous()

        # --- 1. 物理时空流 (Pixel Space) ---
        if self.use_shallow_cnn:
            if self.use_time_cond:
                # ========== 🌟 核心提速与性能突破：Batch 并行 + TSM 交互 ==========
                # 之前是三次独立的 forward，导致 GPU 算力无法吃满，且各帧之间互相隔离
                # 现在把 B 和 T=3 叠起来，变成 [B*3, C, H, W] 一次性轰进去！
                lr_merged = clean_lr_seq.reshape(B * 3, C, H, W)
                
                # 一次性提取特征，在里面经过 15 层 TSM 块时，前中后三帧会自动进行特征交融！
                # 之前：f_merged = self.shallow_feat_extract(lr_merged) # [B*3, 64, H, W]
                # 🌟 显存魔术 1：包裹物理特征提取流
                # use_reentrant=False 是 PyTorch 2.x 的官方推荐规范，性能更好且不易报错
                f_merged = checkpoint(self.shallow_feat_extract, lr_merged, use_reentrant=False)

                # 重新拆分回前、中、后三帧 [B, 3, 64, H, W]
                f_split = f_merged.view(B, 3, 64, H, W)
                f_prev = f_split[:, 0]
                f_curr = f_split[:, 1]
                f_next = f_split[:, 2]
                
                # 1. 基础的通道拼接 (193 通道)
                concat_feat = torch.cat([f_prev, f_curr, f_next, t_map.to(f_prev.dtype)], dim=1)
                
                # 2. 通过 ECA 模块，自适应降权遮挡和严重模糊的通道
                attended_feat = self.eca_layer(concat_feat)
                
                # 3. 将筛选后的高质量特征送入后续的深层残差网络进行融合
                # 之前：fused_spatial = self.time_cond_fusion(attended_feat)
                # 🌟 显存魔术 2：包裹深度残差融合流
                fused_spatial = checkpoint(self.time_cond_fusion, attended_feat, use_reentrant=False)
                # ===========================================================
            else:
                fused_spatial = self.shallow_feat_extract(lr_curr) 
        else:
            fused_spatial = 0.0

        # --- 2. 语义生成流 (SD3 VAE) ---

        if self.use_semantic_prior:
            # ⚠️ 直接使用在方法最开头就已经洗好的 clean_lr_seq，禁止重复调用 pre_cleaner！
            # 将清洗后的数据映射到 VAE 要求的 [-1, 1] 区间
            clean_lr_norm = clean_lr_seq.reshape(B*T, C, H, W) * 2.0 - 1.0

            if pad_h > 0 or pad_w > 0:
                clean_lr_norm = F.pad(clean_lr_norm, (0, pad_w, 0, pad_h), mode='replicate')

            vae_chunk_size = 12
            latents = []

            # 2. 冻结的 VAE 编码过程 (必须严格包裹在 no_grad 中以节省巨量显存)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=False):
                    for i in range(0, B*T, vae_chunk_size):
                        chunk_in = clean_lr_norm[i:i+vae_chunk_size].contiguous()
                        chunk_latent = self.encoder.encode(chunk_in).latent_dist.mode()
                        latents.append(chunk_latent)
                    latent = torch.cat(latents, dim=0)

                # VAE 潜空间标准正态分布漂移校正
                latent = (latent.float() - 0.0609) * 1.5305
                latent = latent.reshape(B, T, 16, H_pad // 8, W_pad // 8)

            # ========== 🌟 终极框架级修复：防止 Checkpoint 梯度饿死 ==========
            # 强行唤醒叶子节点的求导属性，确保后续的 VAE Decoder 必定能收到回传梯度！
            latent.requires_grad_()
            # =============================================================

            # VAE 提取出了未对齐的三帧潜特征
            z_prev, z_curr, z_next = latent[:, 0], latent[:, 1], latent[:, 2]

            if self.use_time_cond:
                # 以当前帧为绝对引力中心，将过去和未来的潜特征拉扯对齐
                z_prev_aligned = self.latent_align(z_curr=z_curr, z_neighbor=z_prev)
                z_next_aligned = self.latent_align(z_curr=z_curr, z_neighbor=z_next)

                # 🌟 显存魔术 3：包裹参数量巨大的 VAE 解码器
                concat_latent = torch.cat([z_prev_aligned, z_curr, z_next_aligned, t_map_latent.to(z_prev.dtype)], dim=1)
                fused_latent = checkpoint(self.vae_decoder, concat_latent, use_reentrant=False)
                # =================================================
            else:
                fused_latent = checkpoint(self.vae_decoder, z_curr, use_reentrant=False)

            if pad_h > 0 or pad_w > 0:
                fused_latent = fused_latent[:, :, :H, :W]

            # ========== 🌟 核心修改：用 SFT 层替换掉原本简单的相加 ==========
            # final_feat = fused_spatial + fused_latent  <-- 删掉或注释掉这行
            final_feat = self.sft_layer(f_physical=fused_spatial, f_semantic=fused_latent)
            # ===============================================================
        else:
            # 消融实验：关闭语义先验时，直接将物理流特征作为最终特征
            final_feat = fused_spatial

        # 🛡️ 铁布衫 1：深层特征极易在 FP16 下冲破 65504，强制把它锁死在安全区！
        final_feat = torch.clamp(final_feat, -10000.0, 10000.0)
        # ===============================================================

        # ========== 新增：计算全局坐标偏移场 ==========
        offset_map = self.offset_conv(final_feat) # [B, 2, H, W]
        
        # 🛡️ 铁布衫 2：绝不能让形变坐标无限大，否则 grid_sample 梯度直接爆炸！
        # 限制最大偏移量为 +/- 40 像素，这对于超分对齐绝对够用了。
        offset_map = torch.clamp(offset_map, -40.0, 40.0)
        # ============================================

        # --- 3. INR 连续推理块分发 ---
        if chunk_size is None or self.training:
            # 记得把 offset_map 也传进 _forward_chunk 里
            return self._forward_chunk(lr_prev, lr_curr, lr_next, final_feat, offset_map, coords_xyt, H, W)
        else:
            B_c, N, _ = coords_xyt.shape
            out_list = []
            for i in range(0, N, chunk_size):
                out_chunk = self._forward_chunk(lr_prev, lr_curr, lr_next, final_feat, offset_map, coords_xyt[:, i:i+chunk_size, :], H, W)
                out_list.append(out_chunk)
            return torch.cat(out_list, dim=1)

    def _forward_chunk(self, lr_prev, lr_curr, lr_next, final_feat, offset_map, coords, H, W):
        # 1. 提取原始基准空间坐标
        spatial_coords = coords[..., :2].unsqueeze(1) # [B, 1, N, 2]
        
        # ========== 新增：Deformable 动态坐标对齐 ==========
        # 🌟 防核爆修复：强制 grid_sample 在 FP32 下运行，防止反向传播梯度溢出
        sampled_offset = F.grid_sample(offset_map.float(), spatial_coords.float(), padding_mode='border', align_corners=False)
        sampled_offset = sampled_offset.to(offset_map.dtype).squeeze(2).permute(0, 2, 1).contiguous() # [B, N, 2]
        
        # ========== 🌟 终极修复 2：物理像素空间分辨率解耦 ==========
        # 训练时 LR 图像的大小永远是 64x64。将其换算到当前推理分辨率。
        scale_x = 64.0 / W
        scale_y = 64.0 / H
        scale_tensor = torch.tensor([scale_x, scale_y], device=sampled_offset.device, dtype=sampled_offset.dtype).view(1, 1, 2)
        sampled_offset = sampled_offset * scale_tensor
        # ========================================================

        # 3. 将预测的连续位移叠加到原始坐标上，得到形变后的坐标
        deformed_spatial_coords = spatial_coords + sampled_offset.unsqueeze(1)
        
        # 4. 使用【形变后的坐标】去提取真正物理对齐的特征！(同样强制 FP32)
        aligned_feat = F.grid_sample(final_feat.float(), deformed_spatial_coords.float(), padding_mode='border', align_corners=False) 
        aligned_feat = aligned_feat.to(final_feat.dtype).squeeze(2).permute(0, 2, 1).contiguous() # [B, N, 64]
        # =================================================
        
        # 5. 计算高频位置编码 (用形变后的坐标计算相对位置，让 MLP 知道真正的物理偏移)
        device = coords.device
        # 用变形后的绝对坐标转换回相对坐标 [-1, 1]
        deformed_coords_flat = deformed_spatial_coords.squeeze(1) # [B, N, 2]
        rel_xy = ((deformed_coords_flat + 1.0) * torch.tensor([W / 2.0, H / 2.0], device=device)) % 1.0 * 2.0 - 1.0
        rel_coords = torch.cat([rel_xy, coords[..., 2:3]], dim=-1)
        
        # 送入位置编码
        encoded_coords = self.pe(rel_coords) 
        
        # 6. 送入 INR MLP 推理高频残差
        with torch.amp.autocast('cuda', enabled=False):
            # 融合的是物理对齐后的特征 aligned_feat
            inr_input = torch.cat([aligned_feat.float(), encoded_coords.float()], dim=-1).contiguous()
            pred_residual = self.inr_mlp(inr_input) 

        # ========== 🧠 拯救残差的物理底图混合器 ==========
        if self.use_time_cond:
            t_chunk = coords[..., 2:3]
            w_p = torch.relu(-t_chunk)         
            w_n = torch.relu(t_chunk)          
            w_c = 1.0 - torch.abs(t_chunk)     
            
            # 🌟 防核爆修复：所有的物理底图采样也全部强制 FP32
            base_p = F.grid_sample(lr_prev.float(), spatial_coords.float(), mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            base_c = F.grid_sample(lr_curr.float(), spatial_coords.float(), mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            base_n = F.grid_sample(lr_next.float(), spatial_coords.float(), mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            
            base_rgb = w_p.to(base_p.dtype) * base_p + w_c.to(base_c.dtype) * base_c + w_n.to(base_n.dtype) * base_n
        else:
            base_rgb = F.grid_sample(lr_curr.float(), spatial_coords.float(), mode='bicubic', padding_mode='border', align_corners=False).squeeze(2).permute(0, 2, 1)
            
        return pred_residual + base_rgb


