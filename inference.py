import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from tqdm import tqdm

from models.st_network import ST_VSR_Network
from train import load_dpas_sr_prior

def load_images(img_dir):
    """读取目录下的所有图片并按字母顺序排序"""
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')) + 
                       glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_paths:
        raise ValueError(f"未在 {img_dir} 中找到图片文件！")
    return img_paths

def preprocess_image(img_path):
    """读取并预处理单张图像到 Tensor [3, H, W]"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    return img_tensor

def main():
    parser = argparse.ArgumentParser(description="ST-VSR Video Sequence Inference")
    parser.add_argument('--input_dir', type=str, required=True, help="输入低分辨率(LR)图像序列的文件夹路径")
    parser.add_argument('--output_dir', type=str, default='results/', help="超分后高分辨率(HR)图像的保存路径")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/st_vsr_best.pth', help="训练好的模型权重路径")
    parser.add_argument('--vae_prior', type=str, default='/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors', help="VAE 先验权重路径")
    parser.add_argument('--scale', type=int, default=4, help="超分放大倍率")
    parser.add_argument('--chunk_size', type=int, default=30000, help="INR MLP 坐标块大小，避免显存溢出")
    parser.add_argument('--fps', type=int, default=30, help="合成视频的帧率")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("🚀 正在初始化 ST-VSR 网络...")
    model = ST_VSR_Network().to(device)
    
    # 注入生成先验
    load_dpas_sr_prior(model, args.vae_prior)
    
    # 加载 EMA 断点权重（测试时必须用 EMA 以保证时序极致平滑）
    print(f"📦 正在加载模型权重: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # model.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # 读取测试序列
    img_paths = load_images(args.input_dir)
    num_frames = len(img_paths)
    print(f"📂 发现 {num_frames} 帧低分辨率图像，准备超分放大 {args.scale} 倍...")

    # 读取第一帧以获取分辨率并生成坐标
    first_frame = preprocess_image(img_paths[0])
    _, H_lr, W_lr = first_frame.shape
    H_hr, W_hr = H_lr * args.scale, W_lr * args.scale
    
    print(f"📏 分辨率转换: {W_lr}x{H_lr} -> {W_hr}x{H_hr}")

    # 预计算全局 HR 坐标矩阵 (中心对齐)
    y_coords = (torch.arange(H_hr, dtype=torch.float32) + 0.5) / H_hr * 2.0 - 1.0
    x_coords = (torch.arange(W_hr, dtype=torch.float32) + 0.5) / W_hr * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    t_tensor = torch.full((coords_xy.shape[0], 1), 0.0) # 测试默认生成中心帧 t=0.0
    coords_xyt = torch.cat([coords_xy, t_tensor], dim=-1).unsqueeze(0).to(device) # [1, N, 3]

    # 初始化视频写入器
    video_path = os.path.join(args.output_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (W_hr, H_hr))

    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="VSR 推理中"):
            # 【时序边界填充逻辑】确保始终有 prev, curr, next 三帧
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(num_frames - 1, i + 1)

            # 读取当前滑动窗口的 3 帧
            frame_prev = preprocess_image(img_paths[idx_prev])
            frame_curr = preprocess_image(img_paths[idx_curr])
            frame_next = preprocess_image(img_paths[idx_next])

            # 组装为 [1, 3, C, H_lr, W_lr] 的时空张量
            lr_seq = torch.stack([frame_prev, frame_curr, frame_next], dim=0).unsqueeze(0).to(device)

            # 纯前向分块计算，避免高分率显存炸裂
            pred_rgb_points = model(lr_seq, coords_xyt, chunk_size=args.chunk_size)
            
            # 张量后处理：还原分辨率并截断
            pred_rgb_clamped = torch.clamp(pred_rgb_points, 0.0, 1.0)
            pred_img = pred_rgb_clamped[0].reshape(H_hr, W_hr, 3).cpu().numpy()
            
            # 转为 uint8 及 BGR 格式保存
            pred_img_uint8 = (pred_img * 255.0).round().astype(np.uint8)
            pred_img_bgr = cv2.cvtColor(pred_img_uint8, cv2.COLOR_RGB2BGR)

            # 保存单帧图片
            frame_name = os.path.basename(img_paths[i])
            cv2.imwrite(os.path.join(args.output_dir, frame_name), pred_img_bgr)
            
            # 写入视频流
            video_writer.write(pred_img_bgr)

    video_writer.release()
    print(f"\n🎉 推理完成！")
    print(f"图像序列已保存至: {args.output_dir}")
    print(f"合成视频已保存至: {video_path}")

if __name__ == "__main__":
    main()