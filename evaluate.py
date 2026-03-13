import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========== 【核心修复 1：欺骗老旧 clip 库，解决 pkg_resources 报错】 ==========
import sys
import types
import packaging.version
mock_pkg_resources = types.ModuleType('pkg_resources')
mock_pkg_resources.packaging = packaging
sys.modules['pkg_resources'] = mock_pkg_resources

# ========== 【核心修复 2：兼容 NumPy 2.x，解决 imgaug 报错】 ==========
import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void]
    }
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
# ========================================================================

import glob
import argparse
import cv2
import torch
import lpips
import pyiqa
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import warnings

warnings.filterwarnings("ignore")

def get_image_paths(folder):
    paths = sorted(glob.glob(os.path.join(folder, '*.png')) + 
                   glob.glob(os.path.join(folder, '*.jpg')))
    return paths

def rgb2y(img):
    img = img.astype(np.float32)
    y = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return np.round(y).clip(0, 255).astype(np.uint8)

def calculate_metrics(pred_path, gt_path, lpips_fn, dists_fn, niqe_fn, musiq_fn, clipiqa_fn, device, crop_border=0):
    pred_img = cv2.imread(pred_path)
    gt_img = cv2.imread(gt_path)

    if pred_img is None or gt_img is None:
        raise ValueError(f"无法读取图片: {pred_path} 或 {gt_path}")

    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    if crop_border > 0:
        pred_img = pred_img[crop_border:-crop_border, crop_border:-crop_border, :]
        gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

    # 1. PSNR 和 SSIM (Y 通道)
    pred_y = rgb2y(pred_img)
    gt_y = rgb2y(gt_img)
    
    psnr_val = psnr_metric(gt_y, pred_y, data_range=255)
    ssim_val = ssim_metric(gt_y, pred_y, data_range=255)

    # 2. 深度感知指标 (RGB 空间)
    pred_tensor_01 = torch.from_numpy(pred_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    gt_tensor_01 = torch.from_numpy(gt_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    pred_tensor_11 = pred_tensor_01 * 2.0 - 1.0
    gt_tensor_11 = gt_tensor_01 * 2.0 - 1.0

    with torch.no_grad():
        lpips_val = lpips_fn(pred_tensor_11.to(device), gt_tensor_11.to(device)).item()
        dists_val = dists_fn(pred_tensor_01.to(device), gt_tensor_01.to(device)).item()
        niqe_val = niqe_fn(pred_tensor_01).item() # CPU 运行
        # 💡 核心修复：直接传入 CPU 张量
        musiq_val = musiq_fn(pred_tensor_01).item()
        clipiqa_val = clipiqa_fn(pred_tensor_01).item()

    return psnr_val, ssim_val, lpips_val, dists_val, niqe_val, musiq_val, clipiqa_val


def main():
    parser = argparse.ArgumentParser(description="VSR 自动化 7 维指标测评脚本")
    parser.add_argument('--pred_dir', type=str, required=True, help="你的模型生成的 HR 图像根目录")
    parser.add_argument('--gt_dir', type=str, required=True, help="Ground Truth 高清原图根目录")
    parser.add_argument('--crop_border', type=int, default=4, help="评估时裁剪的边缘像素")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 正在加载评估模型 (LPIPS, DISTS, NIQE, MUSIQ, CLIPIQA)...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    dists_metric = pyiqa.create_metric('dists', device=device).eval()
    
    cpu_device = torch.device('cpu')
    niqe_metric = pyiqa.create_metric('niqe', device=cpu_device).eval()
    
    # 💡 核心修复：将基于 Transformer 的模型也移至 CPU，避开 V100 矩阵乘法 Bug
    musiq_metric = pyiqa.create_metric('musiq', device=cpu_device).eval()
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=cpu_device).eval()

    seqs = sorted(os.listdir(args.gt_dir))
    seqs = [s for s in seqs if os.path.isdir(os.path.join(args.gt_dir, s))]
    
    if not seqs:
        seqs = ['']
        args.gt_dir = os.path.dirname(args.gt_dir)
        args.pred_dir = os.path.dirname(args.pred_dir)
        seqs = [os.path.basename(args.gt_dir)]

    all_psnr, all_ssim, all_lpips, all_dists, all_niqe, all_musiq, all_clipiqa = [], [], [], [], [], [], []
    total_frames = 0

    print(f"\n📊 测评开始: 边缘裁剪(Crop Border) = {args.crop_border} 像素")
    print("-" * 115)
    # 表头新增了 MUSIQ 和 CLIPIQA
    print(f"{'Sequence':<12} | {'Frames':<6} | {'PSNR(Y)↑':<8} | {'SSIM(Y)↑':<8} | {'LPIPS↓':<8} | {'DISTS↓':<8} | {'NIQE↓':<8} | {'MUSIQ↑':<8} | {'CLIPIQA↑':<8}")
    print("-" * 115)

    for seq in seqs:
        gt_seq_dir = os.path.join(args.gt_dir, seq)
        pred_seq_dir = os.path.join(args.pred_dir, seq)

        gt_paths = get_image_paths(gt_seq_dir)
        pred_paths = get_image_paths(pred_seq_dir)

        if len(gt_paths) == 0: continue

        seq_psnr, seq_ssim, seq_lpips, seq_dists, seq_niqe, seq_musiq, seq_clipiqa = [], [], [], [], [], [], []
        
        min_frames = min(len(gt_paths), len(pred_paths))
        for i in range(min_frames):
            gt_p, pred_p = gt_paths[i], pred_paths[i]
            
            # 接收 7 个返回值
            p_val, s_val, l_val, d_val, n_val, m_val, c_val = calculate_metrics(
                pred_p, gt_p, loss_fn_vgg, dists_metric, niqe_metric, musiq_metric, clipiqa_metric, device, args.crop_border
            )
            
            seq_psnr.append(p_val)
            seq_ssim.append(s_val)
            seq_lpips.append(l_val)
            seq_dists.append(d_val)
            seq_niqe.append(n_val)
            seq_musiq.append(m_val)
            seq_clipiqa.append(c_val)

        avg_p, avg_s = np.mean(seq_psnr), np.mean(seq_ssim)
        avg_l, avg_d, avg_n = np.mean(seq_lpips), np.mean(seq_dists), np.mean(seq_niqe)
        avg_m, avg_c = np.mean(seq_musiq), np.mean(seq_clipiqa)

        all_psnr.extend(seq_psnr)
        all_ssim.extend(seq_ssim)
        all_lpips.extend(seq_lpips)
        all_dists.extend(seq_dists)
        all_niqe.extend(seq_niqe)
        all_musiq.extend(seq_musiq)
        all_clipiqa.extend(seq_clipiqa)
        total_frames += min_frames

        seq_name = seq if seq else 'Single'
        # 打印单行序列结果
        print(f"{seq_name:<12} | {min_frames:<6} | {avg_p:8.2f} | {avg_s:8.4f} | {avg_l:8.4f} | {avg_d:8.4f} | {avg_n:8.4f} | {avg_m:8.4f} | {avg_c:8.4f}")

    print("-" * 115)
    # 打印全局加权平均
    print(f"{'OVERALL':<12} | {total_frames:<6} | {np.mean(all_psnr):8.2f} | {np.mean(all_ssim):8.4f} | {np.mean(all_lpips):8.4f} | {np.mean(all_dists):8.4f} | {np.mean(all_niqe):8.4f} | {np.mean(all_musiq):8.4f} | {np.mean(all_clipiqa):8.4f}")
    print("-" * 115)
    print("💡 指标说明: PSNR/SSIM/MUSIQ/CLIPIQA 越高越好(↑)；LPIPS/DISTS/NIQE 越低越好(↓)。")

if __name__ == '__main__':
    main()