import os
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
        # 彻底移除 random，换成多进程安全的 torch.randint 
        start_idx = int(torch.randint(1, 4, (1,)).item()) 
        input_indices = [start_idx, start_idx + 2, start_idx + 4] # 例: im1, im3, im5
        
        hr_input_frames = []
        for i in input_indices:
            img_path = os.path.join(vid_dir, f'im{i}.png')
            img = cv2.imread(img_path)
            if img is None: # 容错：遇到损坏的图片随机换一个
                return self.__getitem__(int(torch.randint(0, len(self)-1, (1,)).item()))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0 # [3, H, W]
            hr_input_frames.append(img)
            
        _, H_hr, W_hr = hr_input_frames[0].shape
            
        # 3. 随机决定我们要学“超分”(t=0.0) 还是“插帧”(t=-0.5 / 0.5)
        time_choices = [-0.5, 0.0, 0.5]
        t_idx = int(torch.randint(0, 3, (1,)).item()) 
        t_q = time_choices[t_idx]

        # 1. 前面读取完毕，合并为全分辨率 Tensor 
        hr_tensor = torch.stack(hr_input_frames, dim=0) # [3, 3, H, W] 
        
        gt_idx = start_idx + 1 if t_q == -0.5 else (start_idx + 2 if t_q == 0.0 else start_idx + 3) 
        gt_img_path = os.path.join(vid_dir, f'im{gt_idx}.png') 
        gt_img = cv2.imread(gt_img_path) 
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) 
        gt_hr_full = torch.from_numpy(gt_img).float().permute(2, 0, 1) / 255.0 # [3, H, W] 

        # ========== 【🔥 修复：将数据增强提前到全图级别】 ========== 
        if torch.rand(1).item() < 0.5: # 水平翻转 
            hr_tensor = torch.flip(hr_tensor, dims=[3])  # 4D的宽是 dim 3 
            gt_hr_full = torch.flip(gt_hr_full, dims=[2]) # 3D的宽是 dim 2 
            
        if torch.rand(1).item() < 0.5: # 垂直翻转 
            hr_tensor = torch.flip(hr_tensor, dims=[2]) 
            gt_hr_full = torch.flip(gt_hr_full, dims=[1]) 
            
        if torch.rand(1).item() < 0.5: # 时序反转 
            hr_tensor = torch.flip(hr_tensor, dims=[0]) 
            t_q = -t_q 
        # ======================================================== 
        
        # 2. 增强后再下采样 (此时出来的 LR 已经是翻转同步过的) 
        lr_tensor = torch.nn.functional.interpolate(hr_tensor, scale_factor=1/self.scale, mode='bicubic', antialias=True) 
        
        # 3. 随机裁剪 Patch (改用 torch.randint 防止多进程种子冲突) 
        y0 = torch.randint(0, H_hr - self.patch_size + 1, (1,)).item() 
        x0 = torch.randint(0, W_hr - self.patch_size + 1, (1,)).item() 
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

class Vimeo90K_ST_Val_Dataset(Dataset): 
    """ 
    专用于验证集的 Dataloader 
    特点：读取 testlist，不进行随机裁剪，全分辨率评估，固定时间戳预测。 
    新增：采用“微型验证集”策略，均匀采样 max_val_samples 个视频片段以极速验证。
    """ 
    def __init__(self, data_root, scale=4, max_val_samples=500): 
        self.scale = scale 
        self.data_root = data_root 
        
        # 读取验证集列表 
        list_file = os.path.join(data_root, 'sep_testlist.txt') 
        with open(list_file, 'r') as f: 
            all_paths = [line.strip() for line in f if line.strip()] 
            
        # ========== 【核心修改：微型验证集均匀采样】 ========== 
        if len(all_paths) > max_val_samples:
            # 计算切片步长，确保涵盖各种运动幅度的视频
            step = len(all_paths) // max_val_samples
            self.video_paths = all_paths[::step][:max_val_samples]
        else:
            self.video_paths = all_paths
        # ======================================================

    def __len__(self): 
        return len(self.video_paths) 

    def __getitem__(self, idx): 
        vid_dir = os.path.join(self.data_root, 'sequences', self.video_paths[idx]) 
        
        # 1. 固定输入采样：取第 2, 4, 6 帧作为 LR 输入 
        input_indices = [2, 4, 6] 
        hr_input_frames = [] 
        for i in input_indices: 
            img_path = os.path.join(vid_dir, f'im{i}.png') 
            img = cv2.imread(img_path) 
            if img is None: 
                return self.__getitem__((idx + 1) % len(self)) # 容错 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0 # [3, H, W] 
            hr_input_frames.append(img) 
            
        # 2. 生成验证集的 LR 
        hr_tensor = torch.stack(hr_input_frames, dim=0) 
        lr_tensor = torch.nn.functional.interpolate(hr_tensor, scale_factor=1/self.scale, mode='bicubic', antialias=True) 
        
        # 3. 验证目标：我们固定评估中心帧(im4)的超分效果 (t=0.0) 
        gt_img_path = os.path.join(vid_dir, 'im4.png') 
        gt_img = cv2.imread(gt_img_path) 
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) 
        gt_hr_full = torch.from_numpy(gt_img).float().permute(2, 0, 1) / 255.0 
        
        _, H_hr, W_hr = gt_hr_full.shape 
        
        # 4. 全分辨率 3D 时空坐标 (无随机裁剪) 
        y_coords = (torch.arange(H_hr, dtype=torch.float32) / (H_hr - 1)) * 2 - 1 
        x_coords = (torch.arange(W_hr, dtype=torch.float32) / (W_hr - 1)) * 2 - 1 
        
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij') 
        
        coords_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) 
        t_tensor = torch.full((coords_xy.shape[0], 1), 0.0) # t固定为0.0 
        coords_xyt = torch.cat([coords_xy, t_tensor], dim=-1) # [N, 3] 
        
        gt_rgb_points = gt_hr_full.reshape(3, -1).permute(1, 0) # [N, 3] 
        
        return lr_tensor, coords_xyt, gt_rgb_points, H_hr, W_hr