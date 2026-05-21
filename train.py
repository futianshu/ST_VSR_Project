import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

# ========== 【🌟 终极修复：将多进程共享内存转移到文件系统，彻底告别 shm 爆满】 ==========
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# ======================================================================================

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips  

from datasets.vimeo90k_st import Vimeo90K_ST_Dataset, Vimeo90K_ST_Val_Dataset
from models.st_network import ST_VSR_Network

from diffusers import StableDiffusion3Pipeline
from utils.util import load_lora_state_dict
from safetensors.torch import load_file 

from skimage.metrics import structural_similarity as ssim_func 
import numpy as np 
import math
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn 
import cv2 
import kornia.augmentation as K

# 🔥 防止 Dataloader 多进程和 OpenCV 内部多线程冲突 
cv2.setNumThreads(0) 
cv2.ocl.setUseOpenCL(False) 

class CharbonnierLoss(torch.nn.Module): 
    def __init__(self, eps=1e-6): 
        super(CharbonnierLoss, self).__init__() 
        self.eps = eps 

    def forward(self, x, y): 
        with torch.amp.autocast('cuda', enabled=False):
            diff = x.float() - y.float()
            # 加上 1e-9 兜底，彻底封死 sqrt(0) 的理论可能
            return torch.mean(torch.sqrt(diff * diff + self.eps + 1e-9))

class PatchGANDiscriminator(torch.nn.Module):
    # ========== 🌟 恢复原版：2D 条件时空判别器 ==========
    # 输入形状要求：[B, 9, H, W] -> 前中后三帧在通道维度拼接
    def __init__(self, in_channels=9, ndf=64):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1), 
            torch.nn.LeakyReLU(0.2, True)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1), 
            torch.nn.InstanceNorm2d(ndf*2),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1), 
            torch.nn.InstanceNorm2d(ndf*4),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1), 
            torch.nn.InstanceNorm2d(ndf*8),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.layer5 = torch.nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x):
        # 🌟 终极防核爆：判别器同样严格在 FP32 下运行
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            return x

class FocalFrequencyLoss(torch.nn.Module):
    # ========== 🌟 核心创新：频域焦点损失 (补齐微观纹理的最后一块拼图) ==========
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha # 焦点因子：动态放大差异极大区域的惩罚权重

    def forward(self, pred, target):
        # 🌟 终极防核爆：傅里叶变换极其脆弱，必须在纯 FP32 空间下严格运行
        with torch.amp.autocast('cuda', enabled=False):
            pred_f32 = pred.float()
            target_f32 = target.float()

            # 计算 2D 离散傅里叶变换 (正交归一化保证能量守恒)
            pred_fft = torch.fft.fft2(pred_f32, norm='ortho')
            target_fft = torch.fft.fft2(target_f32, norm='ortho')

            # 提取频谱的幅度谱 (Amplitude)
            # 加上 1e-8 防止 torch.abs 在原点处求导产生 NaN
            pred_amp = torch.abs(pred_fft) + 1e-8
            target_amp = torch.abs(target_fft) + 1e-8

            # 计算频域 L1 差异
            diff = torch.abs(pred_amp - target_amp)

            # 动态焦点权重 (Focal Weight)：差异越大的高频部分，网络受到的鞭打越狠
            weight = diff ** self.alpha
            weight = weight.detach() # 权重本身不参与梯度回传

            # 计算最终加权频域损失
            loss = torch.mean(weight * diff)

            return loss * self.loss_weight

def load_dpas_sr_prior(model, vae_safetensors_path):
    if os.path.exists(vae_safetensors_path):
        print(f"🚀 正在提取研究二的图像生成先验: {vae_safetensors_path}")
        try:
            # 🔥 弃用易受干扰的 diffusers pipeline 方法，直接读取底层张量字典 
            vae_lora_state_dict = load_file(vae_safetensors_path) 
            
            load_lora_state_dict(vae_lora_state_dict, model.encoder)
            model.encoder.enable_adapters()
            print("✅ 成功注入 DPAS-SR 的 VAE 先验灵魂！")
        except Exception as e:
            print(f"⚠️ 权重加载遇到异常: {e}")
    else:
        print(f"⚠️ 未找到先验权重: {vae_safetensors_path}，将使用未微调的原始 SD3 VAE。")

    for param in model.encoder.parameters():
        param.requires_grad = False
    print("❄️ 已冻结 Encoder 参数，32G 显存绝对安全！")

    # ========================================================================= 
    # 🛠️ V100 终极显存免疫补丁：强制 Linear 层输入连续，彻底解决 cuBLAS 崩溃！ 
    # ========================================================================= 
    def make_inputs_contiguous(module, args): 
        # 拦截输入，如果是 Tensor，强制瞬间重排为连续内存 .contiguous() 
        return tuple(inp.contiguous() if isinstance(inp, torch.Tensor) else inp for inp in args) 

    patched_layers = 0 
    for module in model.encoder.modules(): 
        # 同时拦截原生的 nn.Linear 以及 PEFT 注入的 lora.Linear 
        if isinstance(module, torch.nn.Linear) or "Linear" in type(module).__name__: 
            module.register_forward_pre_hook(make_inputs_contiguous) 
            patched_layers += 1 
            
    print(f"🔧 已为 V100 成功注入 {patched_layers} 个内存连续性拦截补丁！")

# ==========================================
# 💡 实验名称配置 (每次做新消融实验前，只改这里！)
# ==========================================
EXP_NAME = "ablation_wo_time_cond"  # 移除时序条件引导 (证明 t_map 和动态底图对 tOF 的决定性作用)
# EXP_NAME = "ablation_wo_shallow"      # 当前正在跑：移除物理浅层网络 (证明浅层 CNN 对 PSNR 边缘的锚定作用)
# EXP_NAME = "full_model"             # 当前正在跑：满血完全体
# EXP_NAME = "full_model_tmax30"  # 👈 取消注释并使用这个新名字
# EXP_NAME = "Ours_DeformINR_LatentPrior_Ep55"  # 👈 使用这个全新的终极代号
# EXP_NAME = "ablation_wo_semantic_prior"  # w/o 语义先验
# ==========================================

def main():
    import random 
    
    # 固定全局随机种子，确保实验 100% 可复现 
    seed = 42 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f"checkpoints/{EXP_NAME}", exist_ok=True)
    
    # 💡 极其严谨的消融实验路由控制
    if EXP_NAME == "ablation_wo_time_cond":
        model = ST_VSR_Network(use_time_cond=False, use_shallow_cnn=True).to(device)
        print("🧪 当前运行: 移除时序条件引导 (Time-Condition) 的消融实验")
    elif EXP_NAME == "ablation_wo_shallow":
        model = ST_VSR_Network(use_time_cond=True, use_shallow_cnn=False).to(device)
        print("🧪 当前运行: 移除物理流 (Shallow CNN) 的消融实验")
    elif EXP_NAME == "ablation_wo_semantic_prior":
        model = ST_VSR_Network(use_time_cond=True, use_shallow_cnn=True, use_semantic_prior=False).to(device)
        print("🧪 当前运行: 移除 VAE 语义先验 (Semantic Prior) 的消融实验")
    else:
        model = ST_VSR_Network(use_time_cond=True, use_shallow_cnn=True, use_semantic_prior=True).to(device)
        print("🧪 当前运行: 满血完全体 (双流 + 时序条件引导)")

    load_dpas_sr_prior(model, "/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors")
    
    # ========== 🚀 终极杀手锏：洗清所有 FP16 脏权重 ==========
    # 强制将整个模型统一洗回纯 FP32。
    # 只有底层是 100% 纯 FP32，AMP 才能正常且安全地进行混合精度缩放！
    model = model.float()
    # ==========================================================

    # ==============================================================
    # 🚀 1. 数据集初始化与 Loss 实例化 (保持不变)
    # ==============================================================
    vimeo_root = "/home/ubuntu/data/OpenDataLab___Vimeo90K/raw/vimeo_septuplet"
    train_dataset = Vimeo90K_ST_Dataset(data_root=vimeo_root, scale=4, patch_size=256)
    val_dataset = Vimeo90K_ST_Val_Dataset(data_root=vimeo_root, scale=4)
    # 🔥 终极防泄漏/防卡死配置： 
    # 1. 降低 num_workers 到 4（防止瞬时并发撑爆内存） 
    # 2. pin_memory=False （🌟 极其关键！阻断后台线程占用，彻底根除底层 SHM 内存泄漏） 
    # 3. persistent_workers=True （保持通道常开，拒绝每轮反复创建销毁引发的碎片） 
    train_dataloader = DataLoader( 
        train_dataset, 
        batch_size=16,         # 👈 修改 1：降至 8 释放巨大显存空间
        shuffle=True, 
        num_workers=4,        
        pin_memory=False,      # 👈 必须改成 False，否则必定死锁！
        prefetch_factor=2,    # 🚀 提前预取 2 个 Batch
        persistent_workers=False, # 👈 极其关键：改成 False，彻底根除长周期内存泄漏！
        drop_last=True
    ) 
    
    val_dataloader = DataLoader( 
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=False, 
        persistent_workers=True 
    )
    
    # 建议同时开启 cudnn benchmark (因为你裁切的 patch_size 是固定的，这能提速约 10%) 
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    for param in loss_fn_vgg.parameters(): param.requires_grad = False
    charbonnier = CharbonnierLoss().to(device) 

    # ========== 🌟 新增：实例化频域损失 ==========
    loss_fn_ffl = FocalFrequencyLoss(alpha=1.0).to(device)
    # ============================================
    

    # ==============================================================
    # 🚀 SOTA 提速核心：动态 GPU 退化流水线 (On-the-fly Degradation)
    # ==============================================================
    class GPUDegradation(torch.nn.Module):
        def __init__(self, scale=4):
            super().__init__()
            self.scale = scale

            # ========== 🌟 阶段 1：强退化 (用于 Epoch 1-40 打牢物理基础) ==========
            self.blur_strong = K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.2, 2.0), p=0.5)

            # ========== 🌟 阶段 2：弱退化 (用于 Epoch 41-70 激发感知微雕) ==========
            # 模糊概率降到 30%，sigma 上限降到 1.5
            self.blur_weak = K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.2, 1.5), p=0.3)

            self.noise = K.RandomGaussianNoise(mean=0.0, std=10.0/255.0, p=0.5)
            self.jpeg = K.RandomJPEG(jpeg_quality=(60, 95), p=0.5)

        @torch.no_grad() # 退化过程不需要计算梯度，极致省显存
        # 传入当前 epoch 作为判断依据
        def forward(self, hr_seq, current_epoch=1):
            B, T, C, H, W = hr_seq.shape
            x = hr_seq.view(B*T, C, H, W)

            # 🌟 If 分支：判断当前所处阶段
            if current_epoch > 40:
                x = self.blur_weak(x)
            else:
                x = self.blur_strong(x)

            x = F.interpolate(x, scale_factor=1/self.scale, mode='bicubic', antialias=True)
            x = self.noise(x)
            x = self.jpeg(x)

            x = torch.clamp(x, 0.0, 1.0)
            # 恢复时序维度
            return x.view(B, T, C, H // self.scale, W // self.scale)

    gpu_deg_pipeline = GPUDegradation(scale=4).to(device)
    # ==============================================================


    # ==============================================================
    # 🚀 2. 定义优化器
    # ==============================================================
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    
    # ========== 🌟 核心修改 1：总轮数提升至 55 轮 ==========
    # epochs = 55
    epochs = 70
    
    # ========== 🌟 核心修改 2：引入 Linear Warmup + Cosine Annealing ==========
    warmup_epochs = 5
    cosine_epochs = 35  # 第一阶段物理对齐总共 40 轮 (5 + 35)
    
    # 前 5 轮学习率从 0.01倍 (2e-6) 线性爬升到 2e-4，保护初始化为 0 的偏移器和滤网
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    # 后 35 轮按照余弦曲线平滑衰减到谷底
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
    # 将两个调度器在第 5 轮进行硬拼接
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    # =========================================================================
    
    scaler = torch.cuda.amp.GradScaler() 
    
    # ---------------- 🚀 恢复：2D 判别器初始化 ----------------
    # 👈 修改：in_channels 恢复为 9 (代表 3帧 * RGB 3通道)
    discriminator = PatchGANDiscriminator(in_channels=9).to(device) 
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    # 判别器不需要预热，跟着余弦退火即可
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=40, eta_min=1e-6)
    loss_fn_gan = torch.nn.BCEWithLogitsLoss().to(device)
    # -----------------------------------------------------------------

    # ==============================================================
    # 🚀 3. 加载断点权重
    # ==============================================================
    # ========== 🌟 核心修改 3：断点切换节点后移至 40 轮 ==========
    resume_epoch = 40  # 准备从第 40 轮的断点启动对抗微调
    start_epoch = 1
    best_psnr = 0.0
    
    ema_state_dict_cache = None # 缓存 EMA 权重
    
    if resume_epoch > 0:
        # 直接读取第 40 轮彻底夯实了物理底图的存档
        checkpoint_path = f"checkpoints/{EXP_NAME}/st_vsr_epoch_40.pth"
        # checkpoint_path = f"checkpoints/{EXP_NAME}/st_vsr_latest.pth"
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # ========== 🚀 1. 仅加载生成器的纯净权重 ==========
            clean_model_dict = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(clean_model_dict, strict=False)
            
            # ========== 🚀 2. 仅加载 EMA 的历史平滑权重 ==========
            if 'ema_model_state_dict' in checkpoint:
                ema_state_dict_cache = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in checkpoint['ema_model_state_dict'].items()}
                
            # ========== 🌟 终极防核爆：彻底废弃旧的优化器状态 ==========
            # 绝对不要加载 optimizer.load_state_dict(opt_state)！
            # 让 Adam 优化器从零开始积累 VGG 和 GAN 的真实方差，彻底杜绝除以零的核爆！
            # =========================================================
            
            if 'best_psnr' in checkpoint: best_psnr = checkpoint['best_psnr']
            start_epoch = checkpoint['epoch'] + 1
            print(f"\n✅ 成功加载第 40 轮纯净模型权重！优化器已成功重置为白纸状态！")
            
            # ========== 🌟 终极修复：对抗微调专用的极限低学习率 + TTUR ==========
            if start_epoch >= 41:
                print("🔧 正在为对抗微调阶段重置极低学习率 (TTUR 策略)...")
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5e-6
                
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] = 1e-6
                
                # 🌟 延长余弦退火的周期，从 15 改为 30，让学习率平滑下降 30 轮
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=5e-7)
                scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=30, eta_min=1e-7)

                # 🚀 2. 最关键的一步：如果是中途崩溃恢复（比如 49 轮），必须加载进度！
                if start_epoch > 41:
                    print(f"♻️ 检测到中途恢复！正在读取第 {start_epoch-1} 轮的优化器和调度器进度，无缝接轨！")
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

                    # 👇 完美恢复判别器的火眼金睛
                    if 'discriminator_state_dict' in checkpoint:
                        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            # ==============================================================
        else:
            raise FileNotFoundError(f"❌ 严重错误：找不到断点文件 {checkpoint_path}，请检查路径！")

    # ==============================================================
    # 🚀 4. 编译优化与 EMA 实例化
    # ==============================================================
    # 1. 提取最纯净的原始模型引用（剥离 compile 壳） 
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model 
    
    # 2. EMA 只绑定纯净网络，彻底和 compile 解耦！ 
    ema_model = AveragedModel(raw_model, multi_avg_fn=get_ema_multi_avg_fn(0.999)) 
    
    # ========== 【核心修复：动态适配 EMA 权重前缀】 ========== 
    if ema_state_dict_cache is not None: 
        # 获取当前 EMA 模型实际需要的前缀 
        current_ema_keys = list(ema_model.state_dict().keys()) 
        prefix = "" 
        # 避开 'n_averaged'，找真实的子模块来探测前缀 
        sample_key = next((k for k in current_ema_keys if "t_aiem" in k or "inr_mlp" in k), "") 
        if "module._orig_mod." in sample_key: 
            prefix = "module._orig_mod." 
        elif "module." in sample_key: 
            prefix = "module." 
                
        # 将纯净的 cache 重新拼装上前缀加载进去 
        restored_ema_dict = {} 
        for k, v in ema_state_dict_cache.items(): 
            if k == 'n_averaged': 
                restored_ema_dict[k] = v  # 🔥 绝对不加前缀 
            else: 
                restored_ema_dict[prefix + k] = v 
        
        ema_model.load_state_dict(restored_ema_dict, strict=False) 
        print("✅ 成功恢复 EMA 平滑权重历史！") 
    # =======================================================

    # ========== 🚀 终极防线：开训前强制镇压所有 FP16 脏权重 ==========
    model = model.float()
    discriminator = discriminator.float()
    ema_model.module = ema_model.module.float()
    # =================================================================

    print("🔥 真实世界时空联合训练与验证正式开始！")

    # # ==============================================================
    # # 🚨 极限排雷：单 Batch 500 步死亡压力测试 (找 NaN 元凶) 🚨
    # # ==============================================================
    # # 开启 PyTorch 天眼，只要任何一步算出 NaN，程序会瞬间报错并精准定位到哪一行代码！
    # torch.autograd.set_detect_anomaly(True)
    # # print("⚠️ 注意：正在执行单 Batch 压力测试，寻找 NaN 溢出点...")
    
    # model.train()
    # model.encoder.eval()
    
    # # 1. 强行提取第一个 Batch 作为永远的测试品
    # stress_test_batch = next(iter(train_dataloader))
    # lr_seq_test, coords_xyt_test, gt_rgb_points_test = stress_test_batch
    
    # # 用一个临时进度条来观察
    # stress_pbar = tqdm(range(500), desc="Stress Test")
    
    # for step in stress_pbar:
    #     # 2. 强制把学习率拉满，跳过温柔的 Warmup 掩护，直接暴露问题
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 2e-4
            
    #     # 3. 反复喂入同一个 Batch
    #     lr_seq = lr_seq_test.to(device, non_blocking=True)               
    #     coords_xyt = coords_xyt_test.to(device, non_blocking=True)       
    #     gt_rgb_points = gt_rgb_points_test.to(device, non_blocking=True) 

    #     optimizer.zero_grad(set_to_none=True)
        
    #     with torch.amp.autocast('cuda'):
    #         pred_rgb = model(lr_seq, coords_xyt) 
    #         # 阶段 1：只测 L1 Loss 就足够暴露出底层的数学溢出问题了
    #         loss_l1 = charbonnier(pred_rgb, gt_rgb_points)
    #         loss_G = loss_l1
            
    #     scaler.scale(loss_G).backward()
    #     scaler.unscale_(optimizer) 
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
    #     scaler.step(optimizer)
    #     scaler.update()
        
    #     # 4. 监控关键的数值变化
    #     with torch.no_grad():
    #         pred_rgb_clamped = torch.clamp(pred_rgb.float(), 0.0, 1.0) 
    #         mse_loss = F.mse_loss(pred_rgb_clamped, gt_rgb_points)
    #         psnr = 10 * torch.log10(1.0 / (mse_loss + 1e-8))
            
    #     stress_pbar.set_postfix({
    #         'L_G': f"{loss_G.item():.3f}", 
    #         'PSNR': f"{psnr.item():.2f}",
    #         'Scale': f"{scaler.get_scale():.1f}" # 👈 极其重要：盯着这个值，如果狂跌说明已经出现 inf 了
    #     })
        
    # print("✅ 压力测试完成！如果能看到这句话，说明你的模型在数学上绝对安全，没有任何 NaN！")
    # exit() # 测完直接强制退出，不进入后面的正式训练
    # # ==============================================================
    
    # # --- 原本的正式训练循环，暂时被上面的 exit() 挡住 ---
    
    for epoch in range(start_epoch, epochs + 1):
        # # ========== 🌟 核心修复：将对抗学习率硬重置提前到 Epoch 启动前 ==========
        # if epoch > 40:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 1e-5  # 👈 降速：从 5e-5 降为 1e-5
        #     for param_group in optimizer_D.param_groups:
        #         param_group['lr'] = 2e-5  # 👈 降速：从 1e-4 降为 2e-5
        # # =====================================================================

        # ==========================================================
        # 1. 训练阶段 (Train Phase)
        # ==========================================================
        model.train()
        model.encoder.eval()  # 🔥 必须补上这句，强制 VAE 保持绝对休眠！ 
        # 强制宽度为 100（避免换行），每 2 秒才刷新一次终端（大幅降低 I/O 开销）
        train_pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch}/{epochs}", mininterval=2.0)
        
        # 👇 新增：用于累计当前 Epoch 各项 Loss 的总和
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_loss_Adv = 0.0
        epoch_loss_FFL = 0.0

        for step, (hr_seq, coords_xyt, gt_rgb_points) in enumerate(train_pbar):
            # 注意这里接收的是 hr_seq
            hr_seq = hr_seq.to(device, non_blocking=True)               
            coords_xyt = coords_xyt.to(device, non_blocking=True)       
            gt_rgb_points = gt_rgb_points.to(device, non_blocking=True) 

            # ========== 🚀 在 GPU 上光速完成退化！ ==========
            # 传入当前的 epoch，让退化模块自动切换强弱模式
            lr_seq = gpu_deg_pipeline(hr_seq, current_epoch=epoch)
            # ===============================================

            # ==============================================================
            # 🚀 阶段 1：训练生成器 (Generator)
            # ==============================================================
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                pred_rgb = model(lr_seq, coords_xyt) 
                
                # 🛡️ 铁布衫 3：动态 NaN 拦截器！
                # 万一前向传播算出 NaN，立刻强行丢弃这个 Batch，绝对不允许脏梯度污染模型！
                if torch.isnan(pred_rgb).any() or torch.isinf(pred_rgb).any():
                    print(f"\n⚠️ 警告：检测到极端样本 (Step {step}) 导致溢出！已自动丢弃该 Batch 保住模型权重！")
                    optimizer.zero_grad(set_to_none=True)
                    continue  # 直接跳到下一个 Batch

                # 1. 基础重建损失
                loss_l1 = charbonnier(pred_rgb, gt_rgb_points)
                
                B, N, _ = pred_rgb.shape 
                current_patch_size = int(math.sqrt(N))  
                pred_patch = pred_rgb.permute(0, 2, 1).reshape(B, 3, current_patch_size, current_patch_size) 
                gt_patch = gt_rgb_points.permute(0, 2, 1).reshape(B, 3, current_patch_size, current_patch_size)
                
                # ========== 🌟 核心修改 1：平滑截断，防止越界残差污染 VGG 和 GAN ==========
                # 使用 clamp 的同时梯度依然可以反向传播
                pred_patch_clamped = torch.clamp(pred_patch, 0.0, 1.0)
                
                # ========== 🌟 核心修改 2：按阶段按需计算，彻底释放前期显存与算力 ==========
                if epoch <= 40:
                    # 第一阶段：彻底屏蔽 VGG、GAN 以及没必要的插值拼接，不浪费一丝算力
                    loss_G = loss_l1
                    loss_perceptual = torch.tensor(0.0, device=device)
                    loss_G_adv = torch.tensor(0.0, device=device)
                    loss_D = torch.tensor(0.0, device=device) # 用作 log 占位
                    loss_ffl = torch.tensor(0.0, device=device) # 👈 新增占位
                else:
                    # 第二阶段：才真正开启感知、对抗、频域与上下文构建的巨型计算图！
                    
                    lr_prev_up = F.interpolate(lr_seq[:, 0], size=(current_patch_size, current_patch_size), mode='bicubic', align_corners=False)
                    lr_next_up = F.interpolate(lr_seq[:, 2], size=(current_patch_size, current_patch_size), mode='bicubic', align_corners=False)
                    
                    # ========== 🌟 终极防核爆修复：将所有高级 Loss 强行拉回 FP32 结界 ==========
                    with torch.amp.autocast('cuda', enabled=False):
                        pred_f32 = pred_patch_clamped.float()
                        gt_f32 = gt_patch.float()
                        lr_prev_f32 = lr_prev_up.float()
                        lr_next_f32 = lr_next_up.float()

                        # 在通道维度 (dim=1) 拼接 9 通道 (全 FP32)
                        fake_input_for_G = torch.cat([lr_prev_f32, pred_f32, lr_next_f32], dim=1)
                        
                        # 3. 计算所有高级损失 (VGG 内部含有大量指数运算，FP16 极易溢出，必须 FP32)
                        loss_perceptual = loss_fn_vgg((pred_f32 * 2.0 - 1.0), (gt_f32 * 2.0 - 1.0)).mean() 
                        
                        fake_preds_for_G = discriminator(fake_input_for_G)
                        loss_G_adv = loss_fn_gan(fake_preds_for_G, torch.ones_like(fake_preds_for_G))
                        
                        loss_ffl = loss_fn_ffl(pred_f32, gt_f32)

                        # ========== 🌟 终极方案 B：动态权重退火 (Dynamic Weight Annealing) ==========
                        # 限制 progress 在 [0, 1] 之间，防止从 40 轮甚至更早启动时出现负数
                        progress = max(0.0, min(1.0, (epoch - 41) / (70 - 41 + 1e-8)))

                        # L1 权重：从开局的 0.5 极其平滑地衰减到后期的 0.15
                        weight_l1 = 0.5 - 0.35 * progress

                        # Adv 权重：从开局的 0.05 极其平滑地爬升到后期的 0.12
                        weight_adv = 0.05 + 0.07 * progress

                        loss_G = weight_l1 * loss_l1 + 1.0 * loss_perceptual + weight_adv * loss_G_adv + 0.1 * loss_ffl
                        # =========================================================================
                    # =========================================================================
            
            # 计算 PSNR (仅用作监控)
            with torch.no_grad():
                pred_rgb_clamped = torch.clamp(pred_rgb.float(), 0.0, 1.0) 
                mse_loss = F.mse_loss(pred_rgb_clamped, gt_rgb_points)
                psnr = 10 * torch.log10(1.0 / (mse_loss + 1e-8))
                
            scaler.scale(loss_G).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            scaler.step(optimizer)
            
            # ==============================================================
            # 🚀 阶段 2：训练判别器 (Discriminator)
            # ==============================================================
            # ========== 🌟 核心修改 3：判别器仅在 41 轮后真正耗费算力 ==========
            if epoch > 40:
                optimizer_D.zero_grad(set_to_none=True)
                
                # ========== 🌟 终极防核爆修复：判别器也必须全链路 FP32 ==========
                with torch.amp.autocast('cuda', enabled=False):
                    lr_prev_f32 = lr_prev_up.float()
                    lr_next_f32 = lr_next_up.float()
                    gt_f32 = gt_patch.float()

                    real_input = torch.cat([lr_prev_f32, gt_f32, lr_next_f32], dim=1)
                    real_preds = discriminator(real_input.detach())
                    loss_D_real = loss_fn_gan(real_preds, torch.ones_like(real_preds))
                    
                    # fake_input_for_G 已经在生成器阶段转为 FP32 了
                    fake_input_for_D = fake_input_for_G.detach()
                    fake_preds = discriminator(fake_input_for_D)
                    loss_D_fake = loss_fn_gan(fake_preds, torch.zeros_like(fake_preds))
                    
                    loss_D = (loss_D_real + loss_D_fake) / 2.0
                    
                scaler.scale(loss_D).backward()
                scaler.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler.step(optimizer_D)
            
            scaler.update()
            
            # ========== 🌟 核心修改 5：断崖式 EMA 冻结后移至 40 轮 ==========
            # if epoch <= 40:
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model 
            ema_model.update_parameters(raw_model) 
            # ==========================================================
            
            # 👇 新增：在每步结束时累加 Loss
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            epoch_loss_Adv += loss_G_adv.item()
            epoch_loss_FFL += loss_ffl.item()

            current_lr = optimizer.param_groups[0]['lr']
            
            # 👇 修改：进度条只保留最关心的当前 PSNR 和 学习率
            train_pbar.set_postfix({
                'PSNR': f"{psnr.item():.2f}",
                'LR': f"{current_lr:.2e}"
            })
            
            del lr_seq, coords_xyt, gt_rgb_points, pred_rgb
            
        # ==============================================================
        # 🚀 Epoch 结束后：计算平均 Loss 并打印，调度器步进
        # ==============================================================
        # 👇 新增：结算并打印本轮 Epoch 的平均指标
        num_batches = len(train_dataloader)
        print(f"✅ Epoch {epoch} 训练 Loss 总结 | L_G: {epoch_loss_G/num_batches:.4f} | L_D: {epoch_loss_D/num_batches:.4f} | Adv: {epoch_loss_Adv/num_batches:.4f} | FFL: {epoch_loss_FFL/num_batches:.4f}")


        # ========== 🌟 核心修改 6：阶段性控制学习率 ==========
        scheduler.step()
        scheduler_D.step()
        # ==========================================================
        
        # ==========================================================
        # 2. 验证阶段 (Validation Phase)
        # ==========================================================
        # 🌟 修复：无论第几轮，永远只用 EMA 模型进行验证和测试！它能过滤掉 GAN 90% 的脏噪点！
        ema_model.eval()
        
        val_psnr_avg = 0.0
        val_ssim_avg = 0.0
        val_lpips_avg = 0.0 # 🌟 新增 LPIPS 
        with torch.no_grad(): 
            # 🌟 注意这里加了 enumerate 以获取 step 
            for step, (lr_seq, coords_xyt, gt_rgb_points, h_batch, w_batch) in enumerate(tqdm(val_dataloader, desc=f"Val Epoch {epoch}", leave=False)):
            # 加了 leave=False，验证跑完后进度条会自动消失，保持终端清爽
                lr_seq = lr_seq.to(device, non_blocking=True) 
                coords_xyt = coords_xyt.to(device, non_blocking=True) 
                gt_rgb_points = gt_rgb_points.to(device, non_blocking=True) 
                
                # 🚀 验证集推理也必须开启半精度，否则全分辨率下 VAE 必爆显存！
                with torch.amp.autocast('cuda'):
                    pred_rgb = ema_model(lr_seq, coords_xyt, chunk_size=30000) 
                    
                
                # 强制转回 float32 再做 clamp 和指标计算，保证精度
                pred_rgb_clamped = torch.clamp(pred_rgb.float(), 0.0, 1.0) 
                
                B, N, _ = pred_rgb_clamped.shape 
                h, w = int(h_batch[0]), int(w_batch[0]) 
                
                batch_psnr = 0.0 
                batch_lpips = 0.0 # 🌟 新增 
                for b in range(B): 
                    pred_img = pred_rgb_clamped[b].reshape(h, w, 3) 
                    gt_img = gt_rgb_points[b].reshape(h, w, 3) 
                    
                    # 1. 计算 PSNR 
                    mse = F.mse_loss(pred_img, gt_img) 
                    batch_psnr += (10 * torch.log10(1.0 / (mse + 1e-8))).item() 
                    
                    # 2. 计算 SSIM 
                    pred_img_np = pred_img.cpu().numpy() 
                    gt_img_np = gt_img.cpu().numpy() 
                    batch_ssim = ssim_func(gt_img_np, pred_img_np, data_range=1.0, channel_axis=-1) 
                    val_ssim_avg += batch_ssim / B 

                    # 3. 🌟 计算单张 LPIPS (需调整维度为 [1, 3, H, W] 并映射到 [-1, 1]) 
                    pred_norm = pred_img.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0 
                    gt_norm = gt_img.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0 
                    batch_lpips += loss_fn_vgg(pred_norm, gt_norm).item() 
                
                val_psnr_avg += batch_psnr / B 
                val_lpips_avg += batch_lpips / B # 🌟 累加 
                
                # 🌟 自动保存验证集第一批次的第一张图（已修复串台 Bug）
                if step == 0:
                    import torchvision
                    save_dir = f"checkpoints/{EXP_NAME}/val_images"
                    os.makedirs(save_dir, exist_ok=True)
                    # 🌟 强制统一提取 Batch 中的第 0 个样本 (Index 0)！
                    # 1. 提取第 0 个样本的 LR 并双三次插值放大
                    lr_up = F.interpolate(lr_seq[0:1, 1], size=(h, w), mode='bicubic', align_corners=False).squeeze(0).cpu()
                    # 2. 重新从 Tensor 中精确提取第 0 个样本的 Pred 和 GT (避免受 for 循环变量 b 的污染)
                    pred_save = pred_rgb_clamped[0].reshape(h, w, 3).permute(2, 0, 1).cpu()
                    gt_save = gt_rgb_points[0].reshape(h, w, 3).permute(2, 0, 1).cpu()
                    # 拼接：左(Bicubic) | 中(你的预测) | 右(原图 GT)
                    comparison = torch.stack([lr_up, pred_save, gt_save], dim=0)
                    torchvision.utils.save_image(comparison, f"{save_dir}/epoch_{epoch}_compare.png", nrow=3)
        
        val_psnr_avg /= len(val_dataloader)
        val_ssim_avg /= len(val_dataloader) 
        val_lpips_avg /= len(val_dataloader) # 🌟 平均 LPIPS 
        print(f"📊 Epoch {epoch} 验证集平均 PSNR: {val_psnr_avg:.2f} dB, SSIM: {val_ssim_avg:.4f}, LPIPS: {val_lpips_avg:.4f}")
        
        # ==========================================================
        # 3. 保存逻辑：保存 Best Model 和 Latest Model
        # ==========================================================
        # 获取原始未编译的模型引用，避免 _orig_mod 污染 
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model 
        model_save_dict = {k: v for k, v in raw_model.state_dict().items() if 'encoder' not in k} 
        
        # 剥离 EMA 产生的 module. 和 _orig_mod. 前缀，保存绝对纯净的权重 
        ema_save_dict = {} 
        for k, v in ema_model.state_dict().items(): 
            if 'encoder' in k: continue 
            if k == 'n_averaged': 
                ema_save_dict[k] = v  # 🔥 绝对保留 
                continue 
            clean_key = k.replace('module._orig_mod.', '').replace('module.', '') 
            ema_save_dict[clean_key] = v 
        
        # 打包保存所有关键状态 
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model_save_dict,
            'ema_model_state_dict': ema_save_dict,
            # 👇 保存判别器的网络权重
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),  # 🌟 保存判别器的优化器
            'scheduler_state_dict': scheduler.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),  # 🌟 保存判别器的学习率
            'best_psnr': max(best_psnr, val_psnr_avg)
        }
        
        # 策略 A: 保存最新的 Checkpoint (覆盖式，省空间)
        torch.save(checkpoint_dict, f"checkpoints/{EXP_NAME}/st_vsr_latest.pth")
        
        # 策略 B: 如果当前 PSNR 创新高，保存为 Best Model
        if val_psnr_avg > best_psnr:
            best_psnr = val_psnr_avg
            checkpoint_dict['best_psnr'] = best_psnr # 更新 best_psnr
            torch.save(checkpoint_dict, f"checkpoints/{EXP_NAME}/st_vsr_best.pth")
            print(f"🎉 恭喜！Epoch {epoch} 刷新最高记录！Best PSNR: {best_psnr:.2f} dB")
        
        # 策略 C: 每 10 个 Epoch 留一个存档
        # if epoch % 10 == 0:
        torch.save(checkpoint_dict, f"checkpoints/{EXP_NAME}/st_vsr_epoch_{epoch}.pth")

if __name__ == '__main__':
    main()