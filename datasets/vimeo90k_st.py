import os
import random
import torch
from torch.utils.data import Dataset
import cv2

class Vimeo90K_ST_Dataset(Dataset):
    def __init__(self, data_root, scale=4, patch_size=48):
        self.scale = scale
        self.patch_size = patch_size
        self.data_root = data_root
        
        # 读取 sep_trainlist.txt 中所有的视频片段路径 (例如 00001/0001)
        list_file = os.path.join(data_root, 'sep_trainlist.txt')
        with open(list_file, 'r') as f:
            self.video_paths = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vid_dir = os.path.join(self.data_root, 'sequences', self.video_paths[idx])
        
        # 1. 巧妙的输入采样：间隔抽取 3 帧作为 LR 输入
        start_idx = random.randint(1, 3) 
        input_indices = [start_idx, start_idx + 2, start_idx + 4] # 例: im1, im3, im5
        
        hr_input_frames = []
        for i in input_indices:
            img_path = os.path.join(vid_dir, f'im{i}.png')
            img = cv2.imread(img_path)
            if img is None: # 容错：遇到损坏的图片随机换一个
                return self.__getitem__(random.randint(0, len(self)-1))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0 # [3, H, W]
            hr_input_frames.append(img)
            
        _, H_hr, W_hr = hr_input_frames[0].shape
            
        # 2. 模拟盲退化：下采样得到 LR 输入 
        # (💡写论文时请换成你研究一的复杂退化算法，这里为了快速跑通主干，用 bicubic 占位)
        hr_tensor = torch.stack(hr_input_frames, dim=0)
        lr_tensor = torch.nn.functional.interpolate(hr_tensor, scale_factor=1/self.scale, mode='bicubic', antialias=True) 
        
        # 3. 随机决定我们要学“超分”(t=0.0) 还是“插帧”(t=-0.5 / 0.5)
        time_choices = [-0.5, 0.0, 0.5]
        t_q = random.choice(time_choices)
        
        # 确定对应的真实 HR 图像作为 GT
        gt_idx = start_idx + 1 if t_q == -0.5 else (start_idx + 2 if t_q == 0.0 else start_idx + 3)
        gt_img_path = os.path.join(vid_dir, f'im{gt_idx}.png')
        gt_img = cv2.imread(gt_img_path)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_hr_full = torch.from_numpy(gt_img).float().permute(2, 0, 1) / 255.0
        
        # 4. 核心省显存逻辑：随机裁剪空间 Patch 
        y0 = random.randint(0, H_hr - self.patch_size)
        x0 = random.randint(0, W_hr - self.patch_size)
        gt_patch = gt_hr_full[:, y0:y0+self.patch_size, x0:x0+self.patch_size]
        
        # 5. 生成对应的 3D 时空坐标
        y_coords = (torch.arange(H_hr, dtype=torch.float32) / (H_hr - 1)) * 2 - 1
        x_coords = (torch.arange(W_hr, dtype=torch.float32) / (W_hr - 1)) * 2 - 1
        
        patch_y_coords = y_coords[y0:y0+self.patch_size]
        patch_x_coords = x_coords[x0:x0+self.patch_size]
        grid_y, grid_x = torch.meshgrid(patch_y_coords, patch_x_coords, indexing='ij')
        
        coords_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        t_tensor = torch.full((coords_xy.shape[0], 1), t_q) 
        coords_xyt = torch.cat([coords_xy, t_tensor], dim=-1) # [N, 3]
        
        gt_rgb_points = gt_patch.reshape(3, -1).permute(1, 0) # [N, 3]
        
        return lr_tensor, coords_xyt, gt_rgb_points