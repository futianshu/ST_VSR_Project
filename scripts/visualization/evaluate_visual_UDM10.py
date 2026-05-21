import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_frame_path(method_dir, seq_name, frame_name):
    """
    尝试多种文件名格式来查找帧文件。
    我们的模型和GT用4位数字（0000.png），
    部分竞品方法用8位数字（00000000.png）。
    """
    # 直接尝试原始文件名
    direct = os.path.join(method_dir, seq_name, frame_name)
    if os.path.exists(direct):
        return direct

    # 尝试将4位数字帧名转换为8位（如 0000.png -> 00000000.png）
    base, ext = os.path.splitext(frame_name)
    if base.isdigit():
        alt_name = f"{int(base):08d}{ext}"
        alt_path = os.path.join(method_dir, seq_name, alt_name)
        if os.path.exists(alt_path):
            return alt_path

    return None


def create_visual_comparison(seq_name, frame_name, crop_coords, methods_dict, gt_dir, lr_dir, scale, save_path):
    """
    生成用于论文的局部视觉对比切片图，包含LR输入。
    crop_coords: (x, y, width, height) 对应高分辨率GT图像上的坐标
    """
    x, y, w, h = crop_coords

    # 读取Ground Truth
    gt_path = resolve_frame_path(gt_dir, seq_name, frame_name)
    if gt_path is None:
        print(f"找不到GT文件: {os.path.join(gt_dir, seq_name, frame_name)}")
        return

    gt_img = cv2.imread(gt_path)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    gt_patch = gt_img[y:y+h, x:x+w]

    patches_list = [("GT", gt_patch)]

    # 读取并处理LR低分辨率图（最近邻放大以保留像素块感）
    lr_path = resolve_frame_path(lr_dir, seq_name, frame_name)
    if lr_path is not None:
        lr_img = cv2.imread(lr_path)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        lr_x, lr_y = x // scale, y // scale
        lr_w, lr_h = w // scale, h // scale

        lr_patch_small = lr_img[lr_y:lr_y+lr_h, lr_x:lr_x+lr_w]
        lr_patch_enlarged = cv2.resize(lr_patch_small, (w, h), interpolation=cv2.INTER_NEAREST)
        patches_list.append(("LR", lr_patch_enlarged))
    else:
        print(f"警告: 找不到LR文件 {os.path.join(lr_dir, seq_name, frame_name)}")

    # 收集各对比算法的输出切片
    for method_name, method_dir in methods_dict.items():
        img_path = resolve_frame_path(method_dir, seq_name, frame_name)
        if img_path is not None:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            patch = img[y:y+h, x:x+w]
            patches_list.append((method_name, patch))
        else:
            print(f"警告: 找不到 {method_name} 的文件 {os.path.join(method_dir, seq_name, frame_name)}")

    # 动态计算排版网格
    num_patches = len(patches_list)
    cols = 4
    rows = (num_patches + cols - 1) // cols

    fig = plt.figure(figsize=(15, 6))

    # 左侧放置GT大图并画红框
    ax_main = plt.subplot2grid((rows, cols + 2), (0, 0), rowspan=rows, colspan=2)
    ax_main.imshow(gt_img)
    ax_main.axis('off')

    rect = patches.Rectangle((x, y), w, h, linewidth=2.5, edgecolor='red', facecolor='none')
    ax_main.add_patch(rect)

    # 右侧遍历绘制所有切片
    for idx, (name, patch) in enumerate(patches_list):
        r = idx // cols
        c = idx % cols + 2
        ax_patch = plt.subplot2grid((rows, cols + 2), (r, c))
        ax_patch.imshow(patch)
        ax_patch.axis('off')

        color = 'red' if 'Ours' in name else 'black'
        ax_patch.set_title(name, fontsize=14, color=color, pad=4)

        for spine in ax_patch.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"视觉对比图已保存至: {save_path}")


if __name__ == "__main__":
    GT_DIR = "/home/ubuntu/data/UDM10/GT"
    LR_DIR = "/home/ubuntu/data/UDM10/BIx4"

    METHODS = {
        "STAR":         "/home/ubuntu/data/STAR/GT_video/UDM10_png",
        "SCST":         "/home/ubuntu/lib/SCST/outputs/mococtrl_UDM10",
        "DiffVSR":      "/home/ubuntu/lib/DiffVSR/output/UDM10",
        "RealViformer": "/home/ubuntu/lib/RealViformer/results/RealViformer_UDM10",
        "Ours":         "/home/ubuntu/lib/ST_VSR_Project/results/Ours_DeformINR_LatentPrior_Ep55/epoch_60ema/UDM10_Pred",
    }

    # --- 在此修改目标序列、帧和裁剪区域 ---
    # TARGET_SEQ   = "000"
    # TARGET_FRAME = "0000.png"
    # CROP_BOX     = (400, 200, 120, 120)   # (x, y, width, height)，GT分辨率坐标

    # TARGET_SEQ   = "008"
    # TARGET_FRAME = "0000.png"
    # CROP_BOX     = (1060, 120, 120, 120) 

    TARGET_SEQ   = "003"
    TARGET_FRAME = "0000.png"
    CROP_BOX     = (720, 530, 120, 120) 

    SAVE_DIR  = str(PROJECT_ROOT / "outputs" / "visual_comparisons")
    SAVE_FILE = os.path.join(SAVE_DIR, f"UDM10_{TARGET_SEQ}_{TARGET_FRAME}")

    create_visual_comparison(
        seq_name=TARGET_SEQ,
        frame_name=TARGET_FRAME,
        crop_coords=CROP_BOX,
        methods_dict=METHODS,
        gt_dir=GT_DIR,
        lr_dir=LR_DIR,
        scale=4,
        save_path=SAVE_FILE,
    )
