import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def create_visual_comparison(seq_name, frame_name, crop_coords, methods_dict, gt_dir, lr_dir, scale, save_path):
    """
    生成用于论文的局部视觉对比切片图，包含LR输入，去除了原图下方的文件名
    crop_coords: (x, y, width, height) 对应高分辨率图像上的坐标
    """
    x, y, w, h = crop_coords
    
    # 读取Ground Truth原图
    gt_path = os.path.join(gt_dir, seq_name, frame_name)
    if not os.path.exists(gt_path):
        print(f"找不到GT文件: {gt_path}")
        return
        
    gt_img = cv2.imread(gt_path)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    gt_patch = gt_img[y:y+h, x:x+w]
    
    patches_list = [("GT", gt_patch)]
    
    # 读取并处理原始LR低分辨率图
    lr_path = os.path.join(lr_dir, seq_name, frame_name)
    if os.path.exists(lr_path):
        lr_img = cv2.imread(lr_path)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        
        # 坐标除以scale换算到LR空间
        lr_x, lr_y = x // scale, y // scale
        lr_w, lr_h = w // scale, h // scale
        
        lr_patch_small = lr_img[lr_y:lr_y+lr_h, lr_x:lr_x+lr_w]
        
        # 使用最近邻插值放大，保留明显的马赛克像素块
        lr_patch_enlarged = cv2.resize(lr_patch_small, (w, h), interpolation=cv2.INTER_NEAREST)
        patches_list.append(("LR", lr_patch_enlarged))
    else:
        print(f"警告: 找不到LR文件 {lr_path}")

    # 收集其他算法的高分辨率输出切片
    for method_name, method_dir in methods_dict.items():
        img_path = os.path.join(method_dir, seq_name, frame_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            patch = img[y:y+h, x:x+w]
            patches_list.append((method_name, patch))
        else:
            print(f"警告: 找不到 {method_name} 的文件 {img_path}")
    
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
    
    # 右侧遍历绘制所有小切片
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"视觉对比图已保存至: {save_path}")

if __name__ == "__main__":
    GT_DIR = "/home/ubuntu/data/REDS4/train_sharp"
    LR_DIR = "/home/ubuntu/data/REDS4/train_sharp_bicubic/X4"
    
    # 填入你评估过的基线模型输出路径
    METHODS = {
        "STAR": "/home/ubuntu/data/STAR/GT_video/REDS4_png",
        "SCST": "/home/ubuntu/lib/SCST/outputs/mococtrl_REDS4",
        "DiffVSR": "/home/ubuntu/lib/DiffVSR/output/REDS4",
        "RealViformer": "/home/ubuntu/lib/RealViformer/results/RealViformer_REDS4",
        "Ours": "/home/ubuntu/lib/ST_VSR_Project/results/Ours_DeformINR_LatentPrior_Ep55/epoch_60ema/REDS4_Pred"
    }
    
    # TARGET_SEQ = "020"
    # TARGET_FRAME = "00000000.png" 
    # CROP_BOX = (550, 500, 120, 120) 

    # TARGET_SEQ = "011"
    # TARGET_FRAME = "00000000.png" 
    # CROP_BOX = (200, 350, 120, 120) 

    TARGET_SEQ = "011"
    TARGET_FRAME = "00000000.png" 
    CROP_BOX = (1060, 60, 120, 120) 
    
    SAVE_FILE = str(PROJECT_ROOT / "outputs" / "visual_comparisons" / f"REDS4_{TARGET_SEQ}_{TARGET_FRAME}")
    
    create_visual_comparison(
        seq_name=TARGET_SEQ,
        frame_name=TARGET_FRAME,
        crop_coords=CROP_BOX,
        methods_dict=METHODS,
        gt_dir=GT_DIR,
        lr_dir=LR_DIR,
        scale=4,
        save_path=SAVE_FILE
    )
