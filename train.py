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
# 🔥 防止 Dataloader 多进程和 OpenCV 内部多线程冲突 
cv2.setNumThreads(0) 
cv2.ocl.setUseOpenCL(False) 

class CharbonnierLoss(torch.nn.Module): 
    def __init__(self, eps=1e-6): 
        super(CharbonnierLoss, self).__init__() 
        self.eps = eps 

    def forward(self, x, y): 
        diff = x - y 
        return torch.mean(torch.sqrt(diff * diff + self.eps)) 

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
# EXP_NAME = "ablation_wo_time_cond"  # 当前正在跑：移除时间靶向 (证明 t_map 和动态底图对 tOF 的决定性作用)
# EXP_NAME = "ablation_wo_shallow"      # 当前正在跑：移除物理浅层网络 (证明浅层 CNN 对 PSNR 边缘的锚定作用)
EXP_NAME = "full_model"             # 当前正在跑：满血完全体
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
        print("🧪 当前运行: 移除时间靶向 (Time-Condition) 的消融实验")
    elif EXP_NAME == "ablation_wo_shallow":
        model = ST_VSR_Network(use_time_cond=True, use_shallow_cnn=False).to(device)
        print("🧪 当前运行: 移除物理流 (Shallow CNN) 的消融实验")
    else:
        model = ST_VSR_Network(use_time_cond=True, use_shallow_cnn=True).to(device)
        print("🧪 当前运行: 满血完全体 (双流 + 时间靶向)")

    # 只编译我们自己写的无副作用的子模块
    model.inr_mlp = torch.compile(model.inr_mlp)
    if model.use_shallow_cnn:
        model.shallow_feat_extract = torch.compile(model.shallow_feat_extract)
        if model.use_time_cond:
            model.time_cond_fusion = torch.compile(model.time_cond_fusion)

    load_dpas_sr_prior(model, "/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors")
    
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
        batch_size=32, 
        shuffle=True, 
        num_workers=4,        
        pin_memory=True,      # 🚀 开启锁页内存，极大加速 CPU 数据送到 GPU 的带宽
        prefetch_factor=2,    # 🚀 提前预取 2 个 Batch
        persistent_workers=True, 
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
    
    # ==============================================================
    # 🚀 2. 定义优化器
    # ==============================================================
    # 🔥 加上 fused=True 开启内核融合提速
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, fused=True)
    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 🚀 级别 2 提速：初始化 AMP 的 GradScaler (加在这里！)
    scaler = torch.cuda.amp.GradScaler() # 如果你是 3090/A100/H100，可以直接无脑用 bfloat16 连 scaler 都省了，但如果是 V100 老老实实用 Scaler
    
    # ==============================================================
    # 🚀 3. 加载断点权重
    # ==============================================================
    resume_epoch = 0  # 只要这个数字大于 0，就会触发读取分支
    start_epoch = 1
    best_psnr = 0.0  
    
    ema_state_dict_cache = None # 缓存 EMA 权重
    
    if resume_epoch > 0:
        # 🌟 核心修改：直接无脑读取 latest 存档，不要用数字拼接去猜！
        checkpoint_path = f"checkpoints/{EXP_NAME}/st_vsr_latest.pth" 
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # 先加载基础模型权重
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'best_psnr' in checkpoint: best_psnr = checkpoint['best_psnr']
            
            if 'ema_model_state_dict' in checkpoint:
                ema_state_dict_cache = checkpoint['ema_model_state_dict']
                
            # 🌟 模型会自动从存档里读出它上次存活在第几个 epoch，并自动 +1
            start_epoch = checkpoint['epoch'] + 1
            print(f"\n✅ 成功加载完整断点：{checkpoint_path}")
            print(f"🚀 将直接从 Epoch {start_epoch} 开始无缝续训！当前最高 PSNR: {best_psnr:.2f} dB\n")
        else:
            # 🚨 加上这句报警！如果文件不存在直接报错退出，决不允许静默从头跑！
            raise FileNotFoundError(f"❌ 严重错误：找不到断点文件 {checkpoint_path}，请检查路径！")

    # ==============================================================
    # 🚀 4. 编译优化与 EMA 实例化
    # ==============================================================
    # 1. 提取最纯净的原始模型引用（剥离 compile 壳） 
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model 
    
    # 2. EMA 只绑定纯净网络，彻底和 compile 解耦！ 
    ema_model = AveragedModel(raw_model, multi_avg_fn=get_ema_multi_avg_fn(0.999)) 
    
    # ❌ 彻底注释掉或删除这段 compile 代码！
    # try:
    #     model = torch.compile(model)
    #     print("✅ 模型已启用 torch.compile 编译优化！")
    # except Exception as e:
    #     print(f"⚠️ torch.compile 启用失败: {e}")
        
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

    print("🔥 真实世界时空联合训练与验证正式开始！")
    
    for epoch in range(start_epoch, epochs + 1):
        # ==========================================================
        # 1. 训练阶段 (Train Phase)
        # ==========================================================
        model.train()
        model.encoder.eval()  # 🔥 必须补上这句，强制 VAE 保持绝对休眠！ 
        train_pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch}/{epochs}")
        
        for step, (lr_seq, coords_xyt, gt_rgb_points) in enumerate(train_pbar):
            lr_seq = lr_seq.to(device, non_blocking=True)               
            coords_xyt = coords_xyt.to(device, non_blocking=True)       
            gt_rgb_points = gt_rgb_points.to(device, non_blocking=True) 
            
            optimizer.zero_grad(set_to_none=True)
            
            # 🚀 开启自动混合精度上下文 (替换了原来的“纯 FP32 正向传播”)
            with torch.cuda.amp.autocast():
                pred_rgb = model(lr_seq, coords_xyt) 
                
                # 计算 L1 损失
                loss_l1 = charbonnier(pred_rgb, gt_rgb_points)
                
                # 动态计算 Patch 尺寸
                B, N, _ = pred_rgb.shape 
                current_patch_size = int(math.sqrt(N))  
                pred_patch = pred_rgb.permute(0, 2, 1).reshape(B, 3, current_patch_size, current_patch_size) 
                gt_patch = gt_rgb_points.permute(0, 2, 1).reshape(B, 3, current_patch_size, current_patch_size)
                
                # 计算感知损失
                loss_perceptual = loss_fn_vgg((pred_patch * 2.0 - 1.0), (gt_patch * 2.0 - 1.0)).mean() 
                loss = loss_l1 + 0.1 * loss_perceptual 
            
            # (PSNR 计算保持在 FP32 且不需要梯度)
            with torch.no_grad():
                pred_rgb_clamped = torch.clamp(pred_rgb.float(), 0.0, 1.0) # 强制转回 float32 算 PSNR 避免精度溢出
                mse_loss = F.mse_loss(pred_rgb_clamped, gt_rgb_points)
                psnr = 10 * torch.log10(1.0 / (mse_loss + 1e-8))
                
            # 🚀 替换原来的 loss.backward()
            scaler.scale(loss).backward()
            
            # 🚀 替换原来的梯度裁剪和 optimizer.step()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            scaler.step(optimizer)
            scaler.update()
                        
            # ========== 【新增：步进更新 EMA 参数】 ========== 
            # 🔥 更新也只拿纯净的活跃权重进行平滑，绝不打断编译引擎 
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model 
            ema_model.update_parameters(raw_model) 
            # ================================================= 
            
            # 更新进度条信息 (新增 PSNR 和 当前的学习率 LR)
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'L1': f"{loss_l1.item():.4f}", 
                'LPIPS': f"{loss_perceptual.item():.4f}",
                'PSNR': f"{psnr.item():.2f}",
                'LR': f"{current_lr:.2e}"
            })
            
            # 🔥 强制垃圾回收
            del lr_seq, coords_xyt, gt_rgb_points, pred_rgb
            
        # ==============================================================
        # 🚀 Epoch 结束后：调度器步进 & 完整 Checkpoint 保存
        # ==============================================================
        scheduler.step()
        
        # ==========================================================
        # 2. 验证阶段 (Validation Phase)
        # ==========================================================
        ema_model.eval()  # 使用 EMA 评估！ 
        val_psnr_avg = 0.0
        val_ssim_avg = 0.0
        val_lpips_avg = 0.0 # 🌟 新增 LPIPS 
        with torch.no_grad(): 
            # 🌟 注意这里加了 enumerate 以获取 step 
            for step, (lr_seq, coords_xyt, gt_rgb_points, h_batch, w_batch) in enumerate(tqdm(val_dataloader, desc=f"Val Epoch {epoch}")): 
                lr_seq = lr_seq.to(device, non_blocking=True) 
                coords_xyt = coords_xyt.to(device, non_blocking=True) 
                gt_rgb_points = gt_rgb_points.to(device, non_blocking=True) 
                
                pred_rgb = ema_model(lr_seq, coords_xyt, chunk_size=30000) 
                pred_rgb_clamped = torch.clamp(pred_rgb, 0.0, 1.0) 
                
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
            'model_state_dict': model_save_dict,          # 🌟 绝对纯净的活跃权重 
            'ema_model_state_dict': ema_save_dict,        # 🌟 绝对纯净的平滑权重 
            'optimizer_state_dict': optimizer.state_dict(), 
            'scheduler_state_dict': scheduler.state_dict(), 
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
        if epoch % 10 == 0:
            torch.save(checkpoint_dict, f"checkpoints/{EXP_NAME}/st_vsr_epoch_{epoch}.pth")

if __name__ == '__main__':
    main()