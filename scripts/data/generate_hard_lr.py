import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

def add_noise_and_jpeg(img, noise_sigma=30, jpeg_quality=20):
    """施加极端高斯噪声和双重 JPEG 压缩"""
    # 1. 加强高斯噪声
    noise = np.random.normal(0, noise_sigma, img.shape)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 2. 第一次 JPEG 压缩（产生初始块效应）
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', noisy_img, encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    # 3. 二次 JPEG 压缩（叠加块效应，模拟多次转码）
    _, encimg2 = cv2.imencode('.jpg', decimg, encode_param)
    decimg2 = cv2.imdecode(encimg2, cv2.IMREAD_COLOR)

    return decimg2

def main():
    # 替换为你实际的 REDS4 LR 路径
    input_dir = "/home/ubuntu/data/REDS4/train_sharp_bicubic/X4" 
    output_dir = "/home/ubuntu/data/REDS4_Hard"
    
    seqs = os.listdir(input_dir)
    for seq in seqs:
        in_seq_dir = os.path.join(input_dir, seq)
        out_seq_dir = os.path.join(output_dir, seq)
        os.makedirs(out_seq_dir, exist_ok=True)
        
        imgs = sorted(glob.glob(os.path.join(in_seq_dir, '*.png')))
        for img_path in tqdm(imgs, desc=f"处理序列 {seq}"):
            img = cv2.imread(img_path)
            # 施加极端退化 (Sigma=30的噪声, Quality=20的双重JPEG压缩)
            hard_img = add_noise_and_jpeg(img, noise_sigma=30, jpeg_quality=20)
            
            out_path = os.path.join(out_seq_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, hard_img)
            
    print(f"✅ 极限退化测试集生成完毕，保存在: {output_dir}")

if __name__ == '__main__':
    main()