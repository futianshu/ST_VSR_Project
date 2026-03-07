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
    dataset = Vimeo90K_ST_Dataset(data_root=vimeo_root, scale=4, patch_size=48)
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    for param in loss_fn_vgg.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    
    print("🔥 真实世界时空联合训练正式开始！")
    
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for step, (lr_seq, coords_xyt, gt_rgb_points) in enumerate(pbar):
            lr_seq = lr_seq.to(device)               
            coords_xyt = coords_xyt.to(device)       
            gt_rgb_points = gt_rgb_points.to(device) 
            
            optimizer.zero_grad()
            
            # ==============================================================
            # 🚀 剥离花里胡哨的 autocast，纯 FP32 正向传播，绝对稳如老狗！
            # ==============================================================
            pred_rgb = model(lr_seq, coords_xyt) 
            
            loss_l1 = F.l1_loss(pred_rgb, gt_rgb_points)
            
            patch_size = 48
            B, N, _ = pred_rgb.shape
            pred_patch = pred_rgb.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size)
            gt_patch = gt_rgb_points.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size)
            
            loss_perceptual = loss_fn_vgg((pred_patch * 2.0 - 1.0), (gt_patch * 2.0 - 1.0)).mean()
            loss = loss_l1 + 0.1 * loss_perceptual
                
            # ==============================================================
            # 🚀 剥离 GradScaler，纯 FP32 原生计算梯度，彻底消灭底层硬件 Bug！
            # ==============================================================
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'L1': f"{loss_l1.item():.4f}", 'LPIPS': f"{loss_perceptual.item():.4f}"})
            
        save_dict = {k: v for k, v in model.state_dict().items() if 'encoder' not in k}
        torch.save(save_dict, f"checkpoints/st_vsr_epoch_{epoch}.pth")
        print(f"✅ Epoch {epoch} 轻量化模型已保存。")

if __name__ == '__main__':
    main()