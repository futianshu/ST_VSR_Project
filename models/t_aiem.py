import torch
import torch.nn as nn

class T_AIEM(nn.Module):
    """
    Temporal Adaptive Information Extraction Module
    致敬并升维 RZSR 中的 AIEM 思想：
    通过自适应运动感知，动态对齐并融合时序(前后帧)的隐空间特征。
    """
    def __init__(self, channels=16):
        super().__init__()
        # 移除 AdaptiveAvgPool2d，改为卷积直接输出像素级的空间权重 
        self.routing = nn.Sequential( 
            nn.Conv2d(channels * 3, channels, 3, 1, 1), 
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(channels, 3, 3, 1, 1), # 输出通道为3 (prev, curr, next) 
            nn.Softmax(dim=1) 
        ) 
        
        # 2. 动态特征融合层 
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
    def forward(self, f_prev, f_curr, f_next):
        concat_f = torch.cat([f_prev, f_curr, f_next], dim=1) 
        
        # 此时 w 的形状是 [B, 3, H, W]，包含了空间维度的动态权重 
        w = self.routing(concat_f) 
        
        # 像素级时序自适应重标定 
        f_prev_adp = f_prev * w[:, 0:1, :, :] 
        f_curr_adp = f_curr * w[:, 1:2, :, :] 
        f_next_adp = f_next * w[:, 2:3, :, :] 
        
        aligned_concat = torch.cat([f_prev_adp, f_curr_adp, f_next_adp], dim=1) 
        out = self.fusion(aligned_concat) + f_curr 
        return out