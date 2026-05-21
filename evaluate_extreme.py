import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def get_image_paths(folder):
    return sorted(glob.glob(os.path.join(folder, '*.png')) + 
                  glob.glob(os.path.join(folder, '*.jpg')))

def load_image_tensor(path, device):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)

def rgb2y(img):
    img = img.astype(np.float32)
    y = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return np.round(y).clip(0, 255).astype(np.uint8)

def warp_tensor(x, flow):
    """通用的 Warping 函数，支持扭曲图像或光流场"""
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)

    vgrid = grid + flow

    # 归一化到 [-1, 1] 适配 grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1).contiguous()
    return F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)

def evaluate_sequence_extreme(pred_dir, gt_dir, raft_model, device, motion_thresh=10.0, crop_border=4):
    pred_paths = get_image_paths(pred_dir)
    gt_paths = get_image_paths(gt_dir)
    
    min_frames = min(len(pred_paths), len(gt_paths))
    if min_frames < 2: return None

    metrics = {
        'fast_psnr': [], 'fast_ssim': [],
        'occ_psnr': [], 'occ_ssim': [],
        'fast_pixels': 0, 'occ_pixels': 0
    }
    
    for t in range(min_frames - 1):
        gt_t = load_image_tensor(gt_paths[t], device)
        gt_t_next = load_image_tensor(gt_paths[t+1], device)
        
        pred_img = cv2.imread(pred_paths[t])
        gt_img = cv2.imread(gt_paths[t])
        
        if crop_border > 0:
            pred_img = pred_img[crop_border:-crop_border, crop_border:-crop_border, :]
            gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]
            gt_t = gt_t[:, :, crop_border:-crop_border, crop_border:-crop_border]
            gt_t_next = gt_t_next[:, :, crop_border:-crop_border, crop_border:-crop_border]

        _, _, H, W = gt_t.shape
        H_pad, W_pad = ((H + 7) // 8) * 8, ((W + 7) // 8) * 8
        
        gt_t_pad = F.pad(gt_t, (0, W_pad - W, 0, H_pad - H), mode='replicate') * 2.0 - 1.0
        gt_t_next_pad = F.pad(gt_t_next, (0, W_pad - W, 0, H_pad - H), mode='replicate') * 2.0 - 1.0

        with torch.no_grad():
            # 计算前向与后向光流
            flow_fwd = raft_model(gt_t_pad, gt_t_next_pad)[-1][:, :, :H, :W]
            flow_bwd = raft_model(gt_t_next_pad, gt_t_pad)[-1][:, :, :H, :W]

        # ----------------------------------------------------
        # 1. 极端运动掩码 (Fast Motion Mask)
        # ----------------------------------------------------
        flow_mag = torch.sqrt(flow_fwd[:, 0]**2 + flow_fwd[:, 1]**2)
        fast_mask = (flow_mag > motion_thresh).squeeze().cpu().numpy()
        
        # ----------------------------------------------------
        # 2. 遮挡掩码 (Occlusion Mask)
        # 前向-后向一致性检验
        # ----------------------------------------------------
        warped_bwd = warp_tensor(flow_bwd, flow_fwd)
        flow_diff = flow_fwd + warped_bwd
        diff_mag = torch.sqrt(flow_diff[:, 0]**2 + flow_diff[:, 1]**2)
        
        # 一致性阈值 = 0.01 * 运动幅度 + 0.5 像素
        occ_thresh = 0.01 * flow_mag + 0.5
        occ_mask = (diff_mag > occ_thresh).squeeze().cpu().numpy()

        # ----------------------------------------------------
        # 3. 掩码指标计算 (Y 通道)
        # ----------------------------------------------------
        pred_y = rgb2y(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        gt_y = rgb2y(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        
        mse_map = (pred_y.astype(np.float32) - gt_y.astype(np.float32)) ** 2
        _, ssim_map = ssim_metric(gt_y, pred_y, data_range=255, full=True)

        # 计算快速运动区域指标
        if fast_mask.sum() > 0:
            mse_fast = mse_map[fast_mask].mean()
            psnr_fast = 10 * np.log10(255**2 / (mse_fast + 1e-8)) if mse_fast > 0 else 100.0
            ssim_fast = ssim_map[fast_mask].mean()
            metrics['fast_psnr'].append(psnr_fast)
            metrics['fast_ssim'].append(ssim_fast)
            metrics['fast_pixels'] += fast_mask.sum()

        # 计算遮挡区域指标
        if occ_mask.sum() > 0:
            mse_occ = mse_map[occ_mask].mean()
            psnr_occ = 10 * np.log10(255**2 / (mse_occ + 1e-8)) if mse_occ > 0 else 100.0
            ssim_occ = ssim_map[occ_mask].mean()
            metrics['occ_psnr'].append(psnr_occ)
            metrics['occ_ssim'].append(ssim_occ)
            metrics['occ_pixels'] += occ_mask.sum()

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--motion_thresh', type=float, default=10.0, help="极端运动的像素阈值")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()

    seqs = [s for s in sorted(os.listdir(args.gt_dir)) if os.path.isdir(os.path.join(args.gt_dir, s))]
    if not seqs: seqs = ['']

    all_fast_psnr, all_fast_ssim = [], []
    all_occ_psnr, all_occ_ssim = [], []

    print(f"\n🌪️ 极限场景 (极端运动/遮挡) 定量评估开始")
    print(f"{'Sequence':<12} | {'Fast PSNR':<10} | {'Fast SSIM':<10} | {'Occ PSNR':<10} | {'Occ SSIM':<10}")
    print("-" * 65)

    for seq in tqdm(seqs, desc="计算进度", leave=False):
        gt_seq_dir = os.path.join(args.gt_dir, seq) if seq else args.gt_dir
        pred_seq_dir = os.path.join(args.pred_dir, seq) if seq else args.pred_dir

        metrics = evaluate_sequence_extreme(pred_seq_dir, gt_seq_dir, raft_model, device, args.motion_thresh)
        if not metrics: continue

        if metrics['fast_psnr']:
            all_fast_psnr.extend(metrics['fast_psnr'])
            all_fast_ssim.extend(metrics['fast_ssim'])
        if metrics['occ_psnr']:
            all_occ_psnr.extend(metrics['occ_psnr'])
            all_occ_ssim.extend(metrics['occ_ssim'])

        f_p = np.mean(metrics['fast_psnr']) if metrics['fast_psnr'] else 0
        f_s = np.mean(metrics['fast_ssim']) if metrics['fast_ssim'] else 0
        o_p = np.mean(metrics['occ_psnr']) if metrics['occ_psnr'] else 0
        o_s = np.mean(metrics['occ_ssim']) if metrics['occ_ssim'] else 0

        tqdm.write(f"{seq if seq else 'Single':<12} | {f_p:10.2f} | {f_s:10.4f} | {o_p:10.2f} | {o_s:10.4f}")

    print("-" * 65)
    print(f"{'OVERALL':<12} | {np.mean(all_fast_psnr):10.2f} | {np.mean(all_fast_ssim):10.4f} | {np.mean(all_occ_psnr):10.2f} | {np.mean(all_occ_ssim):10.4f}")
    print("-" * 65)

if __name__ == '__main__':
    main()