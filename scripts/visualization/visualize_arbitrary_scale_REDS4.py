import os
import sys
from pathlib import Path

import cv2
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_inference_multi_scales(input_dir, output_base_dir, checkpoint, scales):
    """
    自动调用推理脚本生成不同倍率的结果
    如果你的机器已经跑过某些倍率，可以把这部分注释掉以节省时间
    """
    seq_name = os.path.basename(input_dir)
    for scale in scales:
        out_dir = os.path.join(output_base_dir, f"scale_{scale}", seq_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # 组装命令行指令
        cmd = [
            sys.executable, str(PROJECT_ROOT / "inference.py"),
            "--input_dir", input_dir,
            "--output_dir", out_dir,
            "--checkpoint", checkpoint,
            "--scale", str(scale),
            "--fps", "30"
        ]
        
        print(f"正在执行推理: Scale = {scale}x")
        subprocess.run(cmd, cwd=PROJECT_ROOT)

def create_scale_staircase(seq_name, frame_name, lr_coords, scales, output_base_dir, lr_dir, save_path):
    """
    读取多尺度推理结果并生成阶梯对比图
    lr_coords: (x, y, w, h) 这是在 LR 低分辨率图上的坐标
    """
    lr_x, lr_y, lr_w, lr_h = lr_coords
    patches = []
    labels = []
    
    # 1. 提取LR原图切片 (作为基准参照)
    lr_path = os.path.join(lr_dir, seq_name, frame_name)
    if os.path.exists(lr_path):
        lr_img = cv2.imread(lr_path)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        img_h, img_w = lr_img.shape[:2]
        # 检查裁剪坐标是否越界
        if lr_x >= img_w or lr_y >= img_h:
            print(f"错误: LR_CROP_BOX ({lr_x}, {lr_y}, {lr_w}, {lr_h}) 超出LR图像范围 ({img_w}x{img_h})")
            return
        # 自动截断到图像边界
        lr_x2 = min(lr_x + lr_w, img_w)
        lr_y2 = min(lr_y + lr_h, img_h)
        lr_patch = lr_img[lr_y:lr_y2, lr_x:lr_x2]
        patches.append(lr_patch)
        labels.append(f"LR Input\n({lr_x2-lr_x}x{lr_y2-lr_y})")
    else:
        print(f"警告: 找不到LR文件 {lr_path}")
        return

    # 2. 提取各个倍率的高清切片
    for scale in scales:
        img_path = os.path.join(output_base_dir, f"scale_{scale}", seq_name, frame_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hr_h_img, hr_w_img = img.shape[:2]

            # 根据倍率动态换算高分辨率图上的裁剪坐标
            hr_x = int(lr_x * scale)
            hr_y = int(lr_y * scale)
            hr_w = int(lr_w * scale)
            hr_h = int(lr_h * scale)

            if hr_x >= hr_w_img or hr_y >= hr_h_img:
                print(f"警告: scale={scale}x 的裁剪坐标 ({hr_x}, {hr_y}) 超出输出图像范围 ({hr_w_img}x{hr_h_img})，跳过")
                continue

            hr_patch = img[hr_y:min(hr_y+hr_h, hr_h_img), hr_x:min(hr_x+hr_w, hr_w_img)]
            patches.append(hr_patch)
            labels.append(f"Ours-GAN {scale}x\n({hr_patch.shape[1]}x{hr_patch.shape[0]})")
        else:
            print(f"警告: 找不到输出文件 {img_path}")

    # 3. 排版绘图：左侧为LR全图（带红框标注裁剪位置），右侧小图排成上下两行
    num_patches = len(patches)
    import math
    cols_right = math.ceil(num_patches / 2)  # 右侧列数
    cols_total = 2 + cols_right               # 左侧全图占2列

    fig = plt.figure(figsize=(3 * cols_total, 7))

    # 左侧：LR全图，跨2行2列
    ax_main = plt.subplot2grid((2, cols_total), (0, 0), rowspan=2, colspan=2)
    ax_main.imshow(lr_img)
    ax_main.axis('off')
    ax_main.set_title("LR Full Image", fontsize=13, color='black', pad=10)

    # 在全图上用红框标出裁剪位置
    rect = mpatches.Rectangle(
        (lr_x, lr_y), lr_w, lr_h,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax_main.add_patch(rect)

    # 右侧：各切片按两行排列（LR crop + 各倍率输出）
    for i in range(num_patches):
        row = i // cols_right
        col = i % cols_right + 2
        ax = plt.subplot2grid((2, cols_total), (row, col))
        # LR切片用最近邻插值保持马赛克，Ours用双三次展示细节
        interp = 'nearest' if i == 0 else 'bicubic'
        ax.imshow(patches[i], interpolation=interp)
        ax.axis('off')

        color = 'red' if 'Ours' in labels[i] else 'black'
        ax.set_title(labels[i], fontsize=13, color=color, pad=10)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"任意比例阶梯展示图已保存至: {save_path}")

if __name__ == "__main__":
    # 配置测试参数
    TARGET_SEQ = "000"
    TARGET_FRAME = "00000000.png"
    LR_CROP_BOX = (200, 50, 32, 32)

    TEST_SCALES = [1.5, 2.5]
    INPUT_SEQ_DIR = f"/home/ubuntu/data/REDS4/train_sharp_bicubic/X4/{TARGET_SEQ}"
    LR_BASE_DIR = "/home/ubuntu/data/REDS4/train_sharp_bicubic/X4"
    OUTPUT_BASE = str(PROJECT_ROOT / "outputs" / "arbitrary_scale_REDS4")
    CHECKPOINT = "/home/ubuntu/lib/ST_VSR_Project/checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_60.pth"

    # 你可以把这个布尔值设为False以跳过推理直接拼图
    NEED_INFERENCE = True
    if NEED_INFERENCE:
        run_inference_multi_scales(INPUT_SEQ_DIR, OUTPUT_BASE, CHECKPOINT, TEST_SCALES)

    frame_stem = os.path.splitext(TARGET_FRAME)[0]
    SAVE_FILE = os.path.join(OUTPUT_BASE, f"arbitrary_scale_REDS4_{TARGET_SEQ}_{frame_stem}.png")
    
    create_scale_staircase(
        seq_name=TARGET_SEQ,
        frame_name=TARGET_FRAME,
        lr_coords=LR_CROP_BOX,
        scales=TEST_SCALES,
        output_base_dir=OUTPUT_BASE,
        lr_dir=LR_BASE_DIR,
        save_path=SAVE_FILE
    )
