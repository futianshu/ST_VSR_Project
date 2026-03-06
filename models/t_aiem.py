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
        # 1. 全局运动感知器 (感知前后帧的运动剧烈程度)
        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(channels // 4, 3, 1), # 动态输出 3 个权重 (对应 prev, curr, next)
            nn.Softmax(dim=1)
        )
        
        # 2. 动态特征融合层 
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
    def forward(self, f_prev, f_curr, f_next):
        # 拼接 3 帧特征
        concat_f = torch.cat([f_prev, f_curr, f_next], dim=1) # [B, C*3, H, W]
        
        # 计算时序动态路由权重 [B, 3, 1, 1]
        w = self.routing(concat_f)
        
        # 时序自适应重标定 (按感知到的运动权重动态叠加，抑制无效拖影)
        f_prev_adp = f_prev * w[:, 0:1]
        f_curr_adp = f_curr * w[:, 1:2]
        f_next_adp = f_next * w[:, 2:3]
        
        # 将对齐后的特征融合，并引入残差强化中心帧的原始画质
        aligned_concat = torch.cat([f_prev_adp, f_curr_adp, f_next_adp], dim=1)
        out = self.fusion(aligned_concat) + f_curr
        return out