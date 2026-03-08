import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips  

from datasets.vimeo90k_st import Vimeo90K_ST_Dataset, Vimeo90K_ST_Val_Dataset
from models.st_network import ST_VSR_Network

from diffusers import StableDiffusion3Pipeline
from utils.util import load_lora_state_dict

from skimage.metrics import structural_similarity as ssim_func 
import numpy as np 
import math
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn 

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
            dir_name = os.path.dirname(vae_safetensors_path)
            file_name = os.path.basename(vae_safetensors_path)
            vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(dir_name, weight_name=file_name)
            
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("checkpoints", exist_ok=True)
    
    model = ST_VSR_Network().to(device)
    load_dpas_sr_prior(model, "/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors")
    
    # ==============================================================
    # 🚀 1. 数据集初始化与 Loss 实例化 (保持不变)
    # ==============================================================
    vimeo_root = "/home/ubuntu/data/OpenDataLab___Vimeo90K/raw/vimeo_septuplet"
    train_dataset = Vimeo90K_ST_Dataset(data_root=vimeo_root, scale=4, patch_size=128)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = Vimeo90K_ST_Val_Dataset(data_root=vimeo_root, scale=4)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    for param in loss_fn_vgg.parameters(): param.requires_grad = False
    charbonnier = CharbonnierLoss().to(device) 
    
    # ==============================================================
    # 🚀 2. 定义优化器
    # ==============================================================
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # ==============================================================
    # 🚀 3. 加载断点权重 (重点：一定要在 Compile 之前加载基础模型权重！)
    # ==============================================================
    resume_epoch = 0 
    start_epoch = 1
    best_psnr = 0.0  
    
    ema_state_dict_cache = None # 缓存 EMA 权重
    
    if resume_epoch > 0:
        checkpoint_path = f"checkpoints/st_vsr_epoch_{resume_epoch}.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # 先加载基础模型权重
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'best_psnr' in checkpoint: best_psnr = checkpoint['best_psnr']
            
            # 将 EMA 权重取出来暂存，等一会儿实例化后再加载
            if 'ema_model_state_dict' in checkpoint:
                ema_state_dict_cache = checkpoint['ema_model_state_dict']
                
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ 成功加载完整断点：{checkpoint_path}，当前最佳 PSNR: {best_psnr:.2f}")

    # ==============================================================
    # 🚀 4. 编译优化与 EMA 实例化
    # ==============================================================
    try:
        model = torch.compile(model)
        print("✅ 模型已启用 torch.compile 编译优化！")
    except Exception as e:
        print(f"⚠️ torch.compile 启用失败: {e}")
        
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)) 
    
    # 现在将缓存的 EMA 权重完美加载进去
    if ema_state_dict_cache is not None:
        ema_model.load_state_dict(ema_state_dict_cache, strict=False)
        print("✅ 成功恢复 EMA 平滑权重历史！")

    print("🔥 真实世界时空联合训练与验证正式开始！")
    
    for epoch in range(start_epoch, epochs + 1):
        # ==========================================================
        # 1. 训练阶段 (Train Phase)
        # ==========================================================
        model.train()
        train_pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch}/{epochs}")
        
        for step, (lr_seq, coords_xyt, gt_rgb_points) in enumerate(train_pbar):
            lr_seq = lr_seq.to(device)               
            coords_xyt = coords_xyt.to(device)       
            gt_rgb_points = gt_rgb_points.to(device) 
            
            optimizer.zero_grad(set_to_none=True)
            
            # ==============================================================
            # 纯 FP32 正向传播
            # ==============================================================
            pred_rgb = model(lr_seq, coords_xyt) 
            
            # 将 F.l1_loss 替换为 charbonnier 
            loss_l1 = charbonnier(pred_rgb, gt_rgb_points)
            
            patch_size = 128  # 必须与 dataset 中的保持一致 
            B, N, _ = pred_rgb.shape
            pred_patch = pred_rgb.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size)
            gt_patch = gt_rgb_points.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size)
            
            # ========== 【修改：彻底阻断感知损失计算图】 ========== 
            if epoch > 10: 
                loss_perceptual = loss_fn_vgg((pred_patch * 2.0 - 1.0), (gt_patch * 2.0 - 1.0)).mean() 
                loss = loss_l1 + 0.1 * loss_perceptual 
            else: 
                # 仅用于日志打印，不参与反向传播 
                loss_perceptual = torch.tensor(0.0, device=device) 
                loss = loss_l1 
            # ====================================================
            
            # 使用截断范围计算训练集 PSNR
            with torch.no_grad():
                pred_rgb_clamped = torch.clamp(pred_rgb, 0.0, 1.0)
                mse_loss = F.mse_loss(pred_rgb_clamped, gt_rgb_points)
                psnr = 10 * torch.log10(1.0 / (mse_loss + 1e-8))
                
            # 反向传播与优化
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()
            
            # ========== 【新增：步进更新 EMA 参数】 ========== 
            ema_model.update_parameters(model) 
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
            
        # ==============================================================
        # 🚀 Epoch 结束后：调度器步进 & 完整 Checkpoint 保存
        # ==============================================================
        scheduler.step()
        
        # ==========================================================
        # 2. 验证阶段 (Validation Phase)
        # ==========================================================
        ema_model.eval()  # 使用 EMA 评估！ 
        val_psnr_avg = 0.0
        val_ssim_avg = 0.0  # 新增 SSIM 
        with torch.no_grad():
            for lr_seq, coords_xyt, gt_rgb_points, h_batch, w_batch in tqdm(val_dataloader, desc=f"Val Epoch {epoch}"):
                lr_seq = lr_seq.to(device)
                coords_xyt = coords_xyt.to(device)
                gt_rgb_points = gt_rgb_points.to(device)
                
                # 传入 chunk_size 进行安全推理 
                pred_rgb = ema_model(lr_seq, coords_xyt, chunk_size=30000)
                
                pred_rgb_clamped = torch.clamp(pred_rgb, 0.0, 1.0)
                mse = F.mse_loss(pred_rgb_clamped, gt_rgb_points)
                val_psnr_avg += 10 * torch.log10(1.0 / (mse + 1e-8))
                
                # ========== 【新增：计算 SSIM】 ========== 
                # SSIM 通常是对每张图单独计算的，这里假设 batch_size 内逐张计算 
                B, N, _ = pred_rgb_clamped.shape 
                # 从 batch 中取出实际的高和宽 (转为 int) 
                h, w = int(h_batch[0]), int(w_batch[0]) 
                
                # 为了简单起见，可以将 batch 中的预测和 GT 还原为 numpy 
                # 由于这是评估，速度要求不苛刻，转到 CPU 计算标准 SSIM 
                # ======================================== 
                # 修复后的 SSIM 张量还原逻辑 
                for b in range(B): 
                    # pred_rgb_clamped[b] 本身是 [h*w, 3] 
                    pred_img = pred_rgb_clamped[b].reshape(h, w, 3).cpu().numpy() 
                    gt_img = gt_rgb_points[b].reshape(h, w, 3).cpu().numpy() 
                    
                    batch_ssim = ssim_func(gt_img, pred_img, data_range=1.0, channel_axis=-1) 
                    val_ssim_avg += batch_ssim / B 
                # ======================================== 
        
        val_psnr_avg /= len(val_dataloader)
        val_ssim_avg /= len(val_dataloader) 
        print(f"📊 Epoch {epoch} 验证集平均 PSNR: {val_psnr_avg:.2f} dB, SSIM: {val_ssim_avg:.4f}")
        
        # ==========================================================
        # 3. 保存逻辑：保存 Best Model 和 Latest Model
        # ==========================================================
        # 排除无需保存的 encoder 参数
        model_save_dict = {k: v for k, v in model.state_dict().items() if 'encoder' not in k}
        ema_save_dict = {k: v for k, v in ema_model.state_dict().items() if 'encoder' not in k} 
        
        # 打包保存所有关键状态
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model_save_dict,          # 活跃权重（用于续训） 
            'ema_model_state_dict': ema_save_dict,        # 🌟 平滑权重（用于推理） 
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': max(best_psnr, val_psnr_avg) # 保存当前的历史最佳
        }
        
        # 策略 A: 保存最新的 Checkpoint (覆盖式，省空间)
        torch.save(checkpoint_dict, "checkpoints/st_vsr_latest.pth")
        
        # 策略 B: 如果当前 PSNR 创新高，保存为 Best Model
        if val_psnr_avg > best_psnr:
            best_psnr = val_psnr_avg
            checkpoint_dict['best_psnr'] = best_psnr # 更新 best_psnr
            torch.save(checkpoint_dict, "checkpoints/st_vsr_best.pth")
            print(f"🎉 恭喜！Epoch {epoch} 刷新最高记录！Best PSNR: {best_psnr:.2f} dB")
        
        # 策略 C: 每 10 个 Epoch 留一个存档
        if epoch % 10 == 0:
            torch.save(checkpoint_dict, f"checkpoints/st_vsr_epoch_{epoch}.pth")

if __name__ == '__main__':
    main()