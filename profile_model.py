import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from thop import profile, clever_format
from models.st_network import ST_VSR_Network

def count_parameters(model):
    """分别统计冻结参数和可训练参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🚀 正在初始化 ST-VSR 架构进行复杂度分析...")
    model = ST_VSR_Network().to(device)
    
    # ========== 【V100 终极显存免疫补丁】 ========== 
    # 强制修复 thop 钩子与 LoRA 钩子冲突导致的 cuBLAS 崩溃
    def make_inputs_contiguous(module, args): 
        return tuple(inp.contiguous() if isinstance(inp, torch.Tensor) else inp for inp in args) 

    for module in model.encoder.modules(): 
        if isinstance(module, torch.nn.Linear) or "Linear" in type(module).__name__: 
            module.register_forward_pre_hook(make_inputs_contiguous) 
    print("🔧 已注入显存连续性补丁，保障 thop 测量稳定运行！")
    # ===============================================

    model.eval()

    # 1. 统计参数量
    total, trainable, frozen = count_parameters(model)
    print("\n" + "="*50)
    print("📊 模型参数量分析 (Parameters)")
    print("="*50)
    print(f"🔹 冻结参数 (SD3 VAE 等): {frozen / 1e6:.2f} M")
    print(f"🔥 可训练参数 (ST_VSR 核心): {trainable / 1e6:.2f} M")
    print(f"📦 总参数量: {total / 1e6:.2f} M")

    # 2. 统计计算量 (FLOPs / MACs)
    print("\n" + "="*50)
    print("⚡ 模型计算复杂度分析 (FLOPs & MACs)")
    print("="*50)
    # 模拟真实 VSR 推理时的输入尺寸：LR 3帧 64x64，放大 4 倍
    dummy_B, dummy_T, dummy_C, dummy_H, dummy_W = 1, 3, 3, 64, 64
    lr_seq = torch.randn(dummy_B, dummy_T, dummy_C, dummy_H, dummy_W).to(device)
    
    scale = 4
    HR_H, HR_W = dummy_H * scale, dummy_W * scale
    N_points = HR_H * HR_W
    coords_xyt = torch.randn(dummy_B, N_points, 3).to(device)

    print(f"📐 设定输入尺寸: LR {dummy_T}帧 {dummy_W}x{dummy_H} -> HR {HR_W}x{HR_H}")
    
    # 使用 thop 计算
    with torch.no_grad():
        macs, params = profile(model, inputs=(lr_seq, coords_xyt, 30000), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.2f")

    print(f"🧮 MACs (乘加累积操作数): {macs_str} (约为 {macs / 1e9:.2f} G MACs)")
    print(f"🧮 FLOPs (浮点运算数)  : 约 {macs * 2 / 1e9:.2f} G FLOPs")
    print("💡 结论提示: 论文中可强调'额外可训练参数极少'以及'基于坐标分块查询带来的低显存占用'。")

if __name__ == '__main__':
    main()