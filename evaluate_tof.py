import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def get_image_paths(folder):
    paths = sorted(glob.glob(os.path.join(folder, '*.png')) + 
                   glob.glob(os.path.join(folder, '*.jpg')))
    return paths

def load_image_tensor(path, device):
    """读取图片并归一化为 [1, 3, H, W] 的 Float Tensor (范围 0~1)"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)

def warp_frame(x, flow):
    """
    使用光流对图像进行 Warping 变换
    x: [B, C, H, W] 待扭曲的图像 (如 Pred_{t-1})
    flow: [B, 2, H, W] 从 t 指向 t-1 的光流
    """
    B, C, H, W = x.size()
    # 生成网格坐标
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)

    # 施加光流形变
    vgrid = grid + flow

    # 归一化到 [-1, 1] 以适配 grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1).contiguous() # [B, H, W, 2]
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
    return output

def calculate_tof(pred_dir, gt_dir, raft_model, device, crop_border=4):
    """计算单个视频序列的时序一致性误差 tOF"""
    pred_paths = get_image_paths(pred_dir)
    gt_paths = get_image_paths(gt_dir)
    
    min_frames = min(len(pred_paths), len(gt_paths))
    if min_frames < 2:
        return 0.0, min_frames

    seq_tof = []
    
    for t in range(1, min_frames):
        # 1. 加载当前帧(t)和上一帧(t-1)
        gt_t = load_image_tensor(gt_paths[t], device)
        gt_t_minus_1 = load_image_tensor(gt_paths[t-1], device)
        
        pred_t = load_image_tensor(pred_paths[t], device)
        pred_t_minus_1 = load_image_tensor(pred_paths[t-1], device)
        
        # 边界裁剪 (保证与空域指标评估范围一致)
        if crop_border > 0:
            gt_t = gt_t[:, :, crop_border:-crop_border, crop_border:-crop_border]
            gt_t_minus_1 = gt_t_minus_1[:, :, crop_border:-crop_border, crop_border:-crop_border]
            pred_t = pred_t[:, :, crop_border:-crop_border, crop_border:-crop_border]
            pred_t_minus_1 = pred_t_minus_1[:, :, crop_border:-crop_border, crop_border:-crop_border]

        # 2. 提取极其精准的光流 (使用 GT 原图提取以保证绝对的物理运动轨迹)
        # RAFT 模型期望输入范围是 [-1, 1] 且分辨率需为 8 的倍数
        _, _, H, W = gt_t.shape
        H_pad = ((H + 7) // 8) * 8
        W_pad = ((W + 7) // 8) * 8
        
        gt_t_pad = F.pad(gt_t, (0, W_pad - W, 0, H_pad - H), mode='replicate') * 2.0 - 1.0
        gt_t_minus_1_pad = F.pad(gt_t_minus_1, (0, W_pad - W, 0, H_pad - H), mode='replicate') * 2.0 - 1.0

        with torch.no_grad():
            # 计算从 t 到 t-1 的光流
            list_of_flows = raft_model(gt_t_pad, gt_t_minus_1_pad)
            flow_t_to_t1 = list_of_flows[-1][:, :, :H, :W] # 去除 Padding 恢复原尺寸

        # 3. 计算 tOF (Temporal Warping Error)
        # 逻辑：把上一帧的预测图，按照真实的物理运动轨迹“拉”到现在的位置
        pred_warped = warp_frame(pred_t_minus_1, flow_t_to_t1)
        
        # 计算预测图 t 与拉扯过来的 t-1 的绝对差异（MSE误差）
        # 学术界惯例：将 MSE 乘以 100 以获得直观的小数表达 (例如 0.12)
        diff = (pred_t - pred_warped) ** 2
        tof_error = diff.mean().item() * 100.0
        
        seq_tof.append(tof_error)

    return np.mean(seq_tof), min_frames

def main():
    parser = argparse.ArgumentParser(description="VSR 时序一致性 tOF 测评脚本")
    parser.add_argument('--pred_dir', type=str, required=True, help="你的模型生成的 HR 图像根目录")
    parser.add_argument('--gt_dir', type=str, required=True, help="Ground Truth 高清原图根目录")
    parser.add_argument('--crop_border', type=int, default=4, help="评估时裁剪的边缘像素")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 正在加载预训练光学流网络 (RAFT)...")
    # 使用 RAFT 网络提取精密光流，参数非常轻量但精度极高
    weights = Raft_Small_Weights.DEFAULT
    raft_model = raft_small(weights=weights, progress=False).to(device).eval()

    seqs = sorted(os.listdir(args.gt_dir))
    seqs = [s for s in seqs if os.path.isdir(os.path.join(args.gt_dir, s))]
    
    if not seqs:
        seqs = ['']
        args.gt_dir = os.path.dirname(args.gt_dir)
        args.pred_dir = os.path.dirname(args.pred_dir)
        seqs = [os.path.basename(args.gt_dir)]

    all_tof = []
    total_pairs = 0

    print(f"\n📊 时序一致性 (tOF) 测评开始: 边缘裁剪 = {args.crop_border} 像素")
    print("-" * 50)
    print(f"{'Sequence':<15} | {'Pairs':<6} | {'tOF (MSE x100) ↓':<15}")
    print("-" * 50)

    for seq in tqdm(seqs, desc="计算进度"):
        gt_seq_dir = os.path.join(args.gt_dir, seq)
        pred_seq_dir = os.path.join(args.pred_dir, seq)

        avg_tof, frames = calculate_tof(pred_seq_dir, gt_seq_dir, raft_model, device, args.crop_border)
        
        if frames > 1:
            all_tof.append(avg_tof * (frames - 1)) # 用于加权平均
            total_pairs += (frames - 1)
            
            seq_name = seq if seq else 'Single'
            tqdm.write(f"{seq_name:<15} | {frames-1:<6} | {avg_tof:10.4f}")

    print("-" * 50)
    overall_tof = np.sum(all_tof) / total_pairs if total_pairs > 0 else 0
    print(f"{'OVERALL':<15} | {total_pairs:<6} | {overall_tof:10.4f}")
    print("-" * 50)
    print("💡 指标说明: tOF 越低越好。数值越低，代表视频生成的帧间连续性越好，闪烁越少。")

if __name__ == '__main__':
    main()