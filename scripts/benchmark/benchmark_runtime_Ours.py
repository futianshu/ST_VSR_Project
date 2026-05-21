import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path

import torch
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.st_network import ST_VSR_Network

def benchmark_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在使用设备: {torch.cuda.get_device_name(device)}")

    # 1. 实例化完整版模型
    # 只测速，不需要加载 .pth 权重，随机初始化的权重耗时是一样的
    model = ST_VSR_Network(use_time_cond=True, use_shallow_cnn=True, use_semantic_prior=True).to(device)
    model.eval()
    model.float() # 保持底层为 FP32，与你 train.py 的设定严格一致

    # 2. 构造严格对齐的测试输入
    # ⚠️ 极其重要：你需要根据你复现基线模型时的设定，修改这里的 LR 分辨率！
    # 假设基准测试是 64x64 放大 4 倍到 256x256
    B, T, C = 1, 3, 3
    H_lr, W_lr = 64, 64  
    scale = 4
    H_hr, W_hr = H_lr * scale, W_lr * scale
    
    # 构造低分辨率输入序列
    dummy_lr_seq = torch.rand(B, T, C, H_lr, W_lr).to(device)

    # 完美复刻你 vimeo90k_st.py 中的评估坐标生成逻辑
    y_coords = (torch.arange(H_hr, dtype=torch.float32) + 0.5) / H_hr * 2.0 - 1.0
    x_coords = (torch.arange(W_hr, dtype=torch.float32) + 0.5) / W_hr * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    t_tensor = torch.full((coords_xy.shape[0], 1), 0.0) # 目标 t=0.0
    dummy_coords_xyt = torch.cat([coords_xy, t_tensor], dim=-1).unsqueeze(0).to(device) # [1, N, 3]

    chunk_size = 30000 # 保持与你实验设计中 5.5.1 节一致的分块大小

    # V100 兼容性：对所有 nn.Linear 注册 pre-hook，强制输入张量连续
    def make_contiguous_hook(_module, args):
        return tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_pre_hook(make_contiguous_hook)

    # 3. GPU 预热 (Warm-up)
    print("🔥 正在唤醒 GPU 并进行预热 (10次前向传播)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_lr_seq, dummy_coords_xyt, chunk_size=chunk_size)
    torch.cuda.synchronize() # 确保预热全部完成

    # 4. 正式测速
    iterations = 50
    print(f"⏱️ 开始正式测速 (连续循环 {iterations} 次)...")

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_lr_seq, dummy_coords_xyt, chunk_size=chunk_size)
    
    torch.cuda.synchronize() # 强制等待所有 GPU 异步计算完成
    end_time = time.perf_counter()

    # 5. 输出结果
    avg_time = (end_time - start_time) / iterations
    print("-" * 50)
    print(f"✅ 测试完成!")
    print(f"📦 输入尺寸: LR {H_lr}x{W_lr} -> HR {H_hr}x{W_hr} (Scale x{scale})")
    print(f"⚙️ 坐标分块: {chunk_size}")
    print(f"⏱️ 单次推理平均耗时: {avg_time:.4f} 秒 ({avg_time * 1000:.2f} 毫秒)")
    print("-" * 50)

if __name__ == '__main__':
    benchmark_model()
