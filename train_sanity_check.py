import torch
import torch.optim as optim
import torch.nn.functional as F
# from torch.amp import autocast, GradScaler
from datasets.vimeo90k_st import Vimeo90K_ST_Dataset
from models.st_network import ST_VSR_Network

def run_sanity_check():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 实例化网络与数据 (Batch Size 设为 4)
    model = ST_VSR_Network().to(device)
    dataset = Vimeo90K_ST_Dataset()
    lr_tensor, coords_xyt, gt_rgb_points = dataset[0] # 取出第一条数据
    
    # 增加 Batch 维度
    lr_tensor = lr_tensor.unsqueeze(0).repeat(4, 1, 1, 1, 1).to(device).contiguous()
    coords_xyt = coords_xyt.unsqueeze(0).repeat(4, 1, 1).to(device).contiguous()
    gt_rgb_points = gt_rgb_points.unsqueeze(0).repeat(4, 1, 1).to(device).contiguous()
    
    # 【显存秘诀】：冻结你从第二篇论文继承来的 Encoder！
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    # scaler = GradScaler('cuda') # 混合精度神器，再省一半显存
    
    print("开始 Sanity Check...")
    model.train()
    
    # 拿着同一批数据死磕 500 步
    for step in range(500):
        optimizer.zero_grad()
        
        # with autocast('cuda'):
        pred_rgb = model(lr_tensor, coords_xyt)
        loss = F.l1_loss(pred_rgb, gt_rgb_points)
        
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        if step % 50 == 0:
            print(f"Step [{step:3d}], Loss: {loss.item():.6f}")

if __name__ == '__main__':
    run_sanity_check()