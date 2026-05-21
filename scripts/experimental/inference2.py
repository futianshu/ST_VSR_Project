import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path

import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from safetensors.torch import load_file 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.st_network import ST_VSR_Network
from utils.util import load_lora_state_dict

def load_dpas_sr_prior(model, vae_safetensors_path):
    if os.path.exists(vae_safetensors_path):
        print(f"🚀 正在提取图像生成先验: {vae_safetensors_path}")
        try:
            vae_lora_state_dict = load_file(vae_safetensors_path) 
            load_lora_state_dict(vae_lora_state_dict, model.encoder)
            model.encoder.enable_adapters()
            print("✅ 成功注入先验灵魂！")
        except Exception as e:
            print(f"⚠️ 权重加载遇到异常: {e}")
    else:
        print(f"⚠️ 未找到先验权重: {vae_safetensors_path}，将使用未微调的原始 SD3 VAE。")

    for param in model.encoder.parameters():
        param.requires_grad = False
    
    def make_inputs_contiguous(module, args): 
        return tuple(inp.contiguous() if isinstance(inp, torch.Tensor) else inp for inp in args) 

    patched_layers = 0 
    for module in model.encoder.modules(): 
        if isinstance(module, torch.nn.Linear) or "Linear" in type(module).__name__: 
            module.register_forward_pre_hook(make_inputs_contiguous) 
            patched_layers += 1 
            
    print(f"🔧 已注入 {patched_layers} 个连续性补丁！")

def load_images(img_dir):
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')) + 
                       glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_paths:
        raise ValueError(f"未在 {img_dir} 中找到图片文件！")
    return img_paths

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    return img_tensor

def main():
    parser = argparse.ArgumentParser(description="ST-VSR Iterative Sharpness Fusion Inference")
    parser.add_argument('--input_dir', type=str, required=True, help="输入 LR 序列文件夹路径")
    parser.add_argument('--output_dir', type=str, default='results/', help="保存路径")
    
    # 👇 核心修改：允许输入两个权重路径
    parser.add_argument('--checkpoint_smooth', type=str, default='checkpoints/full_model/st_vsr_epoch_40.pth', help="平滑底盘模型 (必须用 Ep40 的 EMA 权重！)")
    parser.add_argument('--checkpoint_sharp', type=str, default='checkpoints/full_model/st_vsr_epoch_43.pth', help="锐化特征模型 (可用 Ep43-45 的 Base 权重！)")
    
    parser.add_argument('--vae_prior', type=str, default='/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors', help="VAE 先验权重路径")
    parser.add_argument('--scale', type=float, default=4.0, help="超分倍率")
    parser.add_argument('--chunk_size', type=int, default=30000, help="INR chunk 大小")
    parser.add_argument('--fps', type=int, default=30, help="帧率")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("🚀 正在实例化 ST-VSR 网络...")

    # 1. 初始化两个完全相同的模型
    model_smooth = ST_VSR_Network(use_time_cond=True, use_shallow_cnn=True).to(device)
    model_sharp = ST_VSR_Network(use_time_cond=True, use_shallow_cnn=True).to(device)

    # 2. 为两个模型注入生成先验
    load_dpas_sr_prior(model_smooth, args.vae_prior)
    load_dpas_sr_prior(model_sharp, args.vae_prior)
    
    # ====================================================================
    # 💡 3. 加载平滑底盘模型 (Epoch 40 EMA)
    # ====================================================================
    print(f"📦 正在加载平滑底盘权重 (Ep40 EMA): {args.checkpoint_smooth}")
    checkpoint_smooth = torch.load(args.checkpoint_smooth, map_location=device)
    
    # EMA 权重通常存在 ema_model_state_dict 里
    ema_dict = checkpoint_smooth['ema_model_state_dict']
    
    # 巧妙地剥离 module. 和 _orig_mod. 前缀加载
    clean_ema_dict = {}
    for k, v in ema_dict.items():
        if k == 'n_averaged': continue # 跳过这个占位符
        clean_key = k.replace('module._orig_mod.', '').replace('module.', '')
        clean_ema_dict[clean_key] = v
        
    model_smooth.load_state_dict(clean_ema_dict, strict=False)
    
    # ====================================================================
    # ⚔️ 4. 加载锐化特征模型 (Epoch 43 Base)
    # ====================================================================
    print(f"📦 正在加载锐化特征权重 (Ep43 Base): {args.checkpoint_sharp}")
    checkpoint_sharp = torch.load(args.checkpoint_sharp, map_location=device)
    
    # Base 权重存在 model_state_dict 里
    base_dict = checkpoint_sharp['model_state_dict']
    
    # 剥离前缀加载
    clean_base_dict = {}
    for k, v in base_dict.items():
        clean_key = k.replace('module._orig_mod.', '').replace('module.', '')
        clean_base_dict[clean_key] = v
        
    model_sharp.load_state_dict(clean_base_dict, strict=False)

    model_smooth.eval()
    model_sharp.eval()

    # 读取测试序列
    img_paths = load_images(args.input_dir)
    num_frames = len(img_paths)
    print(f"📂 发现 {num_frames} 帧，超分放大 {args.scale} 倍...")

    # 读取第一帧以获取分辨率
    first_frame = preprocess_image(img_paths[0])
    _, H_lr, W_lr = first_frame.shape
    H_hr, W_hr = int(H_lr * args.scale), int(W_lr * args.scale)
    
    print(f"📏 分辨率转换: {W_lr}x{H_lr} -> {W_hr}x{H_hr}")

    # 预计算坐标
    y_coords = (torch.arange(H_hr, dtype=torch.float32) + 0.5) / H_hr * 2.0 - 1.0
    x_coords = (torch.arange(W_hr, dtype=torch.float32) + 0.5) / W_hr * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    t_tensor = torch.full((coords_xy.shape[0], 1), 0.0)
    coords_xyt = torch.cat([coords_xy, t_tensor], dim=-1).unsqueeze(0).to(device)

    # 初始化视频写入器
    video_path = os.path.join(args.output_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (W_hr, H_hr))

    # ⏳ 缓存初始滑动窗口
    print("⏳ 正在缓存初始滑动窗口...")
    frame_prev = preprocess_image(img_paths[0])
    frame_curr = frame_prev
    frame_next = preprocess_image(img_paths[1]) if num_frames > 1 else frame_curr

    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="VSR 推理中"):
            # 组装为时空张量
            lr_seq = torch.stack([frame_prev, frame_curr, frame_next], dim=0).unsqueeze(0).to(device)

            # 开启半精度
            with torch.amp.autocast('cuda'):
                # ========== 🌟 终极融合方案：分别推理，合并特征 ==========
                # 1. 计算平滑物理底盘
                pred_smooth = model_smooth(lr_seq, coords_xyt, chunk_size=args.chunk_size)
                # 2. 计算激进锐化特征
                pred_sharp = model_sharp(lr_seq, coords_xyt, chunk_size=args.chunk_size)
                
                # ⚔️ 学术融合 (0.5+0.5 混合)：将扎实物理和锐利感知完美中和
                # pred_fused = pred_smooth * 0.5 + pred_sharp * 0.5
                pred_fused = pred_smooth * 1 + pred_sharp * 0
            # ========================================================
            
            # 转回 FP32 再进行还原与截断
            pred_rgb_clamped = torch.clamp(pred_fused.float(), 0.0, 1.0)
            pred_img = pred_rgb_clamped[0].reshape(H_hr, W_hr, 3).cpu().numpy()
            
            # 转为 uint8 及 BGR 格式保存
            pred_img_uint8 = (pred_img * 255.0).round().astype(np.uint8)
            pred_img_bgr = cv2.cvtColor(pred_img_uint8, cv2.COLOR_RGB2BGR)

            # 保存单帧图片
            frame_name = os.path.basename(img_paths[i])
            cv2.imwrite(os.path.join(args.output_dir, frame_name), pred_img_bgr)
            
            # 写入视频流
            video_writer.write(pred_img_bgr)

            # 🔄 滑动窗口向前推进
            frame_prev = frame_curr
            frame_curr = frame_next
            if i + 2 < num_frames:
                frame_next = preprocess_image(img_paths[i + 2])
            else:
                frame_next = frame_curr

    video_writer.release()
    print(f"\n🎉 融合推理完成！图像已融化重塑为锐利先锋！路径: {args.output_dir}")

if __name__ == "__main__":
    main()
