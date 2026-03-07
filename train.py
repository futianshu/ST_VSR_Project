import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips  

from datasets.vimeo90k_st import Vimeo90K_ST_Dataset
from models.st_network import ST_VSR_Network

from diffusers import StableDiffusion3Pipeline
from utils.util import load_lora_state_dict

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
    
    vimeo_root = "/home/ubuntu/data/OpenDataLab___Vimeo90K/raw/vimeo_septuplet"
    # 将 patch_size 从 48 提升到 128 
    dataset = Vimeo90K_ST_Dataset(data_root=vimeo_root, scale=4, patch_size=128) 
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    for param in loss_fn_vgg.parameters():
        param.requires_grad = False
    
    # 实例化 Loss 
    charbonnier = CharbonnierLoss().to(device) 
    
    # ==============================================================
    # 🚀 定义优化器和调度器
    # ==============================================================
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # ==============================================================
    # 🚀 断点续训逻辑 (未来如果需要恢复，只需修改 resume_epoch)
    # ==============================================================
    resume_epoch = 0  # 🌟 本次设为 0 从头开始；未来若在第 30 轮断开，改为 30 即可恢复
    start_epoch = 1
    
    if resume_epoch > 0:
        checkpoint_path = f"checkpoints/st_vsr_epoch_{resume_epoch}.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 恢复模型权重
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # 恢复优化器和调度器状态（无损续训的关键）
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ 成功加载完整断点：{checkpoint_path}，准备从 Epoch {start_epoch} 继续！")
        else:
            print(f"⚠️ 警告：未找到 {checkpoint_path}，将从头开始训练！")
    
    print("🔥 真实世界时空联合训练正式开始！")
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for step, (lr_seq, coords_xyt, gt_rgb_points) in enumerate(pbar):
            lr_seq = lr_seq.to(device)               
            coords_xyt = coords_xyt.to(device)       
            gt_rgb_points = gt_rgb_points.to(device) 
            
            optimizer.zero_grad()
            
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
            
            loss_perceptual = loss_fn_vgg((pred_patch * 2.0 - 1.0), (gt_patch * 2.0 - 1.0)).mean()
            loss = loss_l1 + 0.1 * loss_perceptual
            
            # ==============================================================
            # 🚀 计算 PSNR (峰值信噪比)
            # ==============================================================
            mse_loss = F.mse_loss(pred_rgb, gt_rgb_points)
            psnr = 10 * torch.log10(1.0 / (mse_loss + 1e-8))
                
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            # 更新进度条信息 (新增 PSNR 和 当前的学习率 LR)
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
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
        
        # 排除无需保存的 encoder 参数
        model_save_dict = {k: v for k, v in model.state_dict().items() if 'encoder' not in k}
        
        # 打包保存所有关键状态
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model_save_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint_dict, f"checkpoints/st_vsr_epoch_{epoch}.pth")
        print(f"✅ Epoch {epoch} Checkpoint（含模型权重及优化器状态）已保存。")

if __name__ == '__main__':
    main()