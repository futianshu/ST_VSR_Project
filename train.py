import os
# 🌟【第一重防线】：强制引入国内高速镜像源！彻底绕过 Hugging Face 断网限制！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["HF_TOKEN"] = "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"

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips  

from datasets.vimeo90k_st import Vimeo90K_ST_Dataset
from models.st_network import ST_VSR_Network

from diffusers import StableDiffusion3Pipeline
from utils.util import load_lora_state_dict

def load_dpas_sr_prior(model, vae_safetensors_path):
    """提取 DPAS-SR 训好的 VAE 先验"""
    if os.path.exists(vae_safetensors_path):
        print(f"🚀 正在提取研究二的图像生成先验: {vae_safetensors_path}")
        try:
            # 借用 diffusers 的 safetensors 读取逻辑 (参数是目录和文件名)
            dir_name = os.path.dirname(vae_safetensors_path)
            file_name = os.path.basename(vae_safetensors_path)
            
            vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(dir_name, weight_name=file_name)
            
            # 使用你之前工程里的加载函数精确注入权重
            load_lora_state_dict(vae_lora_state_dict, model.encoder)
            model.encoder.enable_adapters()
            print("✅ 成功注入 DPAS-SR 的 VAE 先验灵魂！")
        except Exception as e:
            print(f"⚠️ 权重加载遇到异常: {e}")
    else:
        print(f"⚠️ 未找到先验权重: {vae_safetensors_path}，将使用未微调的原始 SD3 VAE。")

    # 🌟【第二重防线】：彻底冻结 Encoder，保住你 32G 显卡的命脉！
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("❄️ 已冻结 Encoder 参数，32G 显存绝对安全！")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. 初始化模型 (有镜像源加持，这里会自动飞速下载几百MB的VAE)
    model = ST_VSR_Network().to(device)
    
    # 2. 注入你上一篇论文 (DPAS-SR) 跑出来的 Encoder 权重
    load_dpas_sr_prior(model, "/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors")
    
    # 3. 初始化真实数据集
    vimeo_root = "/home/ubuntu/data/OpenDataLab___Vimeo90K/raw/vimeo_septuplet"
    dataset = Vimeo90K_ST_Dataset(data_root=vimeo_root, scale=4, patch_size=48)
    
    # Batch Size 设 8，配合 8 个 workers
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    
    # 4. 损失函数 (L1管整体结构，LPIPS管高频纹理)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    for param in loss_fn_vgg.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    print("🔥 真实世界时空联合训练正式开始！")
    
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for step, (lr_seq, coords_xyt, gt_rgb_points) in enumerate(pbar):
            lr_seq = lr_seq.to(device)               # [B, 3, 3, H_lr, W_lr]
            coords_xyt = coords_xyt.to(device)       # [B, N, 3]
            gt_rgb_points = gt_rgb_points.to(device) # [B, N, 3]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred_rgb = model(lr_seq, coords_xyt) # [B, N, 3]
                
                # --- 算 L1 Loss ---
                loss_l1 = F.l1_loss(pred_rgb, gt_rgb_points)
                
                # --- 算 LPIPS Loss ---
                patch_size = 48
                B, N, _ = pred_rgb.shape
                pred_patch = pred_rgb.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size)
                gt_patch = gt_rgb_points.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size)
                
                # LPIPS 期望输入范围是 [-1, 1]
                loss_perceptual = loss_fn_vgg((pred_patch * 2.0 - 1.0), (gt_patch * 2.0 - 1.0)).mean()
                
                # 总 Loss 加权
                loss = loss_l1 + 0.1 * loss_perceptual
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'L1': f"{loss_l1.item():.4f}", 'LPIPS': f"{loss_perceptual.item():.4f}"})
            
        # 🌟【第三重防线】：剔除庞大的 VAE 权重，只保存我们微创新的 3D 隐式参数！防止服务器几十G硬盘被撑爆！
        save_dict = {k: v for k, v in model.state_dict().items() if 'encoder' not in k}
        torch.save(save_dict, f"checkpoints/st_vsr_epoch_{epoch}.pth")
        print(f"✅ Epoch {epoch} 轻量化模型已保存。")

if __name__ == '__main__':
    main()