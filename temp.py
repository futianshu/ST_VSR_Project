import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips  

# ========== 【修改 1：导入新写的验证集类】 ==========
from datasets.vimeo90k_st import Vimeo90K_ST_Dataset, Vimeo90K_ST_Val_Dataset
from models.st_network import ST_VSR_Network

from diffusers import StableDiffusion3Pipeline
from utils.util import load_lora_state_dict

# ... [保留 load_dpas_sr_prior 函数，不修改] ...

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("checkpoints", exist_ok=True)
    
    model = ST_VSR_Network().to(device)
    load_dpas_sr_prior(model, "/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors")
    
    vimeo_root = "/home/ubuntu/data/OpenDataLab___Vimeo90K/raw/vimeo_septuplet"
    
    # 初始化训练集
    train_dataset = Vimeo90K_ST_Dataset(data_root=vimeo_root, scale=4, patch_size=48)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    
    # ========== 【新增：初始化验证集】 ==========
    # 验证集不需要 shuffle，batch_size 设小一点（比如2或4），因为评估的是全分辨率大图
    val_dataset = Vimeo90K_ST_Val_Dataset(data_root=vimeo_root, scale=4)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    # ==========================================
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    for param in loss_fn_vgg.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    resume_epoch = 0 
    start_epoch = 1
    best_psnr = 0.0  # 🌟 新增：追踪全局最高 PSNR
    
    if resume_epoch > 0:
        checkpoint_path = f"checkpoints/st_vsr_epoch_{resume_epoch}.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复历史最高 PSNR
            if 'best_psnr' in checkpoint:
                best_psnr = checkpoint['best_psnr']
                
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ 成功加载完整断点：{checkpoint_path}，准备从 Epoch {start_epoch} 继续！当前最佳 PSNR: {best_psnr:.2f}")
    
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
            
            optimizer.zero_grad()
            
            pred_rgb = model(lr_seq, coords_xyt) 
            
            loss_l1 = F.l1_loss(pred_rgb, gt_rgb_points)
            
            patch_size = 48
            B, N, _ = pred_rgb.shape
            pred_patch = pred_rgb.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size)
            gt_patch = gt_rgb_points.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size)
            
            loss_perceptual = loss_fn_vgg((pred_patch * 2.0 - 1.0), (gt_patch * 2.0 - 1.0)).mean()
            loss = loss_l1 + 0.1 * loss_perceptual
            
            # 使用截断范围计算训练集 PSNR
            with torch.no_grad():
                pred_rgb_clamped = torch.clamp(pred_rgb, 0.0, 1.0)
                mse_loss = F.mse_loss(pred_rgb_clamped, gt_rgb_points)
                train_psnr = 10 * torch.log10(1.0 / (mse_loss + 1e-8))
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'LPIPS': f"{loss_perceptual.item():.4f}",
                'PSNR': f"{train_psnr.item():.2f}",
                'LR': f"{current_lr:.2e}"
            })
            
        scheduler.step()
        
        # ==========================================================
        # 2. 验证阶段 (Validation Phase) 🌟 新增逻辑
        # ==========================================================
        model.eval()
        val_psnr_sum = 0.0
        print(f"\n⏳ 正在验证 Epoch {epoch} ...")
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Val Epoch {epoch}")
            for lr_seq, coords_xyt, gt_rgb_points in val_pbar:
                lr_seq = lr_seq.to(device)
                coords_xyt = coords_xyt.to(device)
                gt_rgb_points = gt_rgb_points.to(device)
                
                # 纯预测与评估
                pred_rgb = model(lr_seq, coords_xyt)
                pred_rgb_clamped = torch.clamp(pred_rgb, 0.0, 1.0)
                
                mse_loss = F.mse_loss(pred_rgb_clamped, gt_rgb_points)
                batch_psnr = 10 * torch.log10(1.0 / (mse_loss + 1e-8))
                
                val_psnr_sum += batch_psnr.item()
                val_pbar.set_postfix({'Val_PSNR': f"{batch_psnr.item():.2f}"})
                
        avg_val_psnr = val_psnr_sum / len(val_dataloader)
        print(f"📊 Epoch {epoch} 验证集平均 PSNR: {avg_val_psnr:.4f}")
        
        # ==========================================================
        # 3. Checkpoint 保存策略
        # ==========================================================
        model_save_dict = {k: v for k, v in model.state_dict().items() if 'encoder' not in k}
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model_save_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': max(best_psnr, avg_val_psnr) # 保存最佳记录以供恢复
        }
        
        # 保存常规 Epoch 权重
        torch.save(checkpoint_dict, f"checkpoints/st_vsr_epoch_{epoch}.pth")
        
        # 🌟 如果突破了历史最好成绩，额外保存一份 best_model
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(checkpoint_dict, "checkpoints/best_model.pth")
            print(f"🏆 发现新的最佳模型！已更新 best_model.pth (最高 PSNR: {best_psnr:.4f})\n")

if __name__ == '__main__':
    main()