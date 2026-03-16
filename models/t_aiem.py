import torch
import torch.nn as nn

class T_AIEM(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        # 🔥 核心修改：在 Latent 空间使用 5x5 卷积，感受野覆盖原图 40x40 的运动范围
        self.routing = nn.Sequential( 
            nn.Conv2d(channels * 3, channels, 5, 1, 2), # padding=2
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True), 
            nn.Conv2d(channels, 3, 3, 1, 1), 
            nn.Softmax(dim=1) 
        ) 
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
    def forward(self, f_prev, f_curr, f_next):
        concat_f = torch.cat([f_prev, f_curr, f_next], dim=1) 
        w = self.routing(concat_f) 
        
        f_prev_adp = f_prev * w[:, 0:1, :, :] 
        f_curr_adp = f_curr * w[:, 1:2, :, :] 
        f_next_adp = f_next * w[:, 2:3, :, :] 
        
        aligned_concat = torch.cat([f_prev_adp, f_curr_adp, f_next_adp], dim=1) 
        out = self.fusion(aligned_concat) + f_curr 
        return out