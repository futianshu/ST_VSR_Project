"""
参数规模统计脚本
比较 ST-VSR 与 5 个参考模型的参数量
"""
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def count_params(model):
    """统计模型参数量（可训练 + 总计）"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def fmt(n):
    """格式化参数量为 M（百万）"""
    return f"{n/1e6:.2f}M"

results = {}

# ─────────────────────────────────────────────────────────────
# 1. ST-VSR（本文模型）
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("[1/6] ST-VSR (本文模型)")
try:
    sys.path.insert(0, str(PROJECT_ROOT))
    from models.st_network import ST_VSR_Network

    model = ST_VSR_Network(use_time_cond=True, use_shallow_cnn=True, use_semantic_prior=True)
    total, trainable = count_params(model)
    # 注：SD3 VAE 骨干网络被冻结，只有 LoRA 适配器可训练
    # 分别统计各子模块
    sub = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        sub[name] = n
        print(f"  {name}: {fmt(n)}")
    print(f"  ── 总参数: {fmt(total)} | 可训练: {fmt(trainable)}")
    results['ST-VSR (Ours)'] = {'total': total, 'trainable': trainable}
except Exception as e:
    print(f"  ERROR: {e}")
    results['ST-VSR (Ours)'] = None
finally:
    if 'ST_VSR_Network' in dir():
        del model
    # 清理导入，避免命名冲突
    for key in list(sys.modules.keys()):
        if 'st_network' in key or 'models' in key:
            del sys.modules[key]

# ─────────────────────────────────────────────────────────────
# 2. DiffVSR
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("[2/6] DiffVSR")
try:
    _orig_dir = os.getcwd()
    os.chdir('/home/ubuntu/lib/DiffVSR')
    sys.path.insert(0, '/home/ubuntu/lib/DiffVSR')
    from models.unet import UNet3DVSRModel
    import json

    with open('/home/ubuntu/lib/DiffVSR/configs/unet_3d_config.json') as f:
        cfg = json.load(f)

    model = UNet3DVSRModel(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        block_out_channels=tuple(cfg['block_out_channels']),
        layers_per_block=cfg['layers_per_block'],
        cross_attention_dim=cfg['cross_attention_dim'],
        attention_head_dim=cfg['attention_head_dim'],
        down_block_types=cfg['down_block_types'],
        up_block_types=cfg['up_block_types'],
        only_cross_attention=cfg['only_cross_attention'],
        use_linear_projection=cfg['use_linear_projection'],
        num_class_embeds=cfg['num_class_embeds'],
        temporal_module_config=cfg['temporal_module_config'],
        down_temporal_idx=cfg['down_temporal_idx'],
        up_temporal_idx=cfg['up_temporal_idx'],
        mid_temporal=cfg['mid_temporal'],
        msatten=cfg['msatten'],
    )
    total, trainable = count_params(model)
    print(f"  UNet3DVSRModel 总参数: {fmt(total)} | 可训练: {fmt(trainable)}")
    results['DiffVSR'] = {'total': total, 'trainable': trainable}
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    results['DiffVSR'] = None
finally:
    os.chdir(_orig_dir)
    for key in list(sys.modules.keys()):
        if key.startswith('models') and 'unet' in key:
            del sys.modules[key]

# ─────────────────────────────────────────────────────────────
# 3. FMA-Net
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("[3/6] FMA-Net")
try:
    sys.path.insert(0, '/home/ubuntu/lib/FMA-Net')
    from config import Config
    from model import FMANet

    config = Config('/home/ubuntu/lib/FMA-Net/experiment.cfg')
    model = FMANet(config)
    total, trainable = count_params(model)
    # 分别统计两个子网络
    d_params = sum(p.numel() for p in model.degradation_learning_network.parameters())
    if hasattr(model, 'restoration_network'):
        r_params = sum(p.numel() for p in model.restoration_network.parameters())
        print(f"  Net_D (降质学习): {fmt(d_params)}")
        print(f"  Net_R (恢复网络): {fmt(r_params)}")
    print(f"  ── 总参数: {fmt(total)} | 可训练: {fmt(trainable)}")
    results['FMA-Net'] = {'total': total, 'trainable': trainable}
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    results['FMA-Net'] = None
finally:
    for key in list(sys.modules.keys()):
        if key in ('config', 'model'):
            del sys.modules[key]

# ─────────────────────────────────────────────────────────────
# 4. RealViformer
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("[4/6] RealViformer")
try:
    sys.path.insert(0, '/home/ubuntu/lib/RealViformer')
    from archs.realviformer_arch import RealViformer

    model = RealViformer(
        num_feat=48,
        num_blocks=[2, 3, 4, 1],
        spynet_path=None,  # 不加载预训练权重，只统计结构
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        merge_head=2,
        bias=False,
        LayerNorm_type='BiasFree',
        ch_compress=True,
        squeeze_factor=[4, 4, 4],
        masked=True,
    )
    total, trainable = count_params(model)
    print(f"  RealViformer 总参数: {fmt(total)} | 可训练: {fmt(trainable)}")
    results['RealViformer'] = {'total': total, 'trainable': trainable}
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    results['RealViformer'] = None
finally:
    for key in list(sys.modules.keys()):
        if 'realviformer' in key or ('archs' in key):
            del sys.modules[key]

# ─────────────────────────────────────────────────────────────
# 5. SCST
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("[5/6] SCST")
try:
    import importlib.util
    import types

    # ── 构建 basicsr stub，绕过 mmcv 依赖 ──
    def _make_pkg(name):
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__package__ = name
        sys.modules[name] = mod
        return mod

    # 顶层 basicsr 包（不触发 __init__）
    bs = _make_pkg('basicsr')
    bs_utils = _make_pkg('basicsr.utils')
    bs_utils_reg = _make_pkg('basicsr.utils.registry')
    bs_archs = _make_pkg('basicsr.archs')
    bs_archs_util = _make_pkg('basicsr.archs.arch_util')
    bs_archs_spy = _make_pkg('basicsr.archs.spynet_arch')

    # ARCH_REGISTRY stub：假装是一个装饰器工厂
    class _FakeRegistry:
        def register(self, cls=None, name=None):
            if cls is None:
                return lambda c: c
            return cls
    bs_utils_reg.ARCH_REGISTRY = _FakeRegistry()

    # ResidualBlockNoBN / make_layer stub
    class _ResBlock(nn.Module):
        def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        def forward(self, x): return x

    def _make_layer(block, num_blocks, **kwargs):
        return nn.Sequential(*[block(**kwargs) for _ in range(num_blocks)])

    def _flow_warp(x, flow, **kw): return x

    bs_archs_util.ResidualBlockNoBN = _ResBlock
    bs_archs_util.make_layer = _make_layer
    bs_archs_util.flow_warp = _flow_warp

    # SpyNet stub（只统计参数，不实际推理）
    class _SpyNet(nn.Module):
        def __init__(self, pretrained=None):
            super().__init__()
            self.basic_module = nn.ModuleList([
                nn.Sequential(nn.Conv2d(8, 32, 7, 1, 3), nn.ReLU(), nn.Conv2d(32, 16, 7, 1, 3),
                              nn.ReLU(), nn.Conv2d(16, 2, 7, 1, 3))
                for _ in range(6)])
        def forward(self, ref, supp): return torch.zeros(ref.shape[0], 2, *ref.shape[-2:])

    bs_archs_spy.SpyNet = _SpyNet

    # 现在加载 tempo_model_arch
    spec = importlib.util.spec_from_file_location(
        'tempo_model_arch',
        '/home/ubuntu/lib/SCST/basicsr/archs/tempo_model_arch.py'
    )
    tempo_mod = importlib.util.module_from_spec(spec)
    sys.modules['tempo_model_arch'] = tempo_mod
    spec.loader.exec_module(tempo_mod)
    CouplePropModuleWithFlowNet = tempo_mod.CouplePropModuleWithFlowNet

    model = CouplePropModuleWithFlowNet(num_ch=4, num_feat=64, num_block=5)
    total, trainable = count_params(model)
    print(f"  CouplePropModuleWithFlowNet 总参数: {fmt(total)} | 可训练: {fmt(trainable)}")
    results['SCST'] = {'total': total, 'trainable': trainable}
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    results['SCST'] = None
finally:
    for key in list(sys.modules.keys()):
        if 'basicsr' in key or 'tempo_model' in key:
            del sys.modules[key]

# ─────────────────────────────────────────────────────────────
# 6. STAR（VSR 特定组件）
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("[6/6] STAR")
try:
    # 直接用 importlib 加载 VSR 模块，绕过 opensora.__init__（依赖 mmengine）
    import importlib.util

    def load_file(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    safmn_mod = load_file('safmn_arch', '/home/ubuntu/lib/STAR/utils_data/opensora/models/vsr/safmn_arch.py')
    SAFMN = safmn_mod.SAFMN

    # SAFMN (x4 SR backbone)
    safmn = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=4, use_res=True)
    safmn_params = count_params(safmn)
    print(f"  SAFMN (x4 超分主干): {fmt(safmn_params[0])}")

    # FDIE 中的两个小 SAFMN
    safmn1 = SAFMN(dim=72, n_blocks=8, ffn_scale=2.0, upscaling_factor=1, in_dim=6, use_res=True)
    safmn2 = SAFMN(dim=72, n_blocks=8, ffn_scale=2.0, upscaling_factor=1, in_dim=6, use_res=True)
    print(f"  SAFMN-HF: {fmt(count_params(safmn1)[0])}")
    print(f"  SAFMN-LF: {fmt(count_params(safmn2)[0])}")

    # SFR-LFTG（inline，绕过 xformers 依赖）
    class SpatialFeatureRefiner(nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.hf_linear = nn.Linear(hidden_channels, hidden_channels * 2)
            self.lf_linear = nn.Linear(hidden_channels, hidden_channels * 2)
            self.gelu = nn.GELU()
            self.fusion_linear = nn.Linear(hidden_channels * 2, hidden_channels)

    class LFTemporalGuider(nn.Module):
        def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
            super().__init__()
            self.q_linear = nn.Linear(d_model, d_model)
            self.kv_linear = nn.Linear(d_model, d_model * 2)
            self.proj = nn.Linear(d_model, d_model)
    sfr = SpatialFeatureRefiner(hidden_channels=3072)
    lftg = LFTemporalGuider(d_model=3072, num_heads=48)
    sfr_params = count_params(sfr)[0]
    lftg_params = count_params(lftg)[0]
    print(f"  SFR (空间特征精炼): {fmt(sfr_params)}")
    print(f"  LFTG (低频时序引导): {fmt(lftg_params)}")

    star_vsr_total = safmn_params[0] + count_params(safmn1)[0] + count_params(safmn2)[0] + sfr_params + lftg_params
    print(f"  ── VSR 附加模块合计: {fmt(star_vsr_total)}")
    print(f"  注：STAR 主干为 CogVideoX-5B（~50亿参数），上述仅为 VSR 特有模块")
    results['STAR (VSR modules)'] = {'total': star_vsr_total, 'trainable': star_vsr_total}
    results['STAR (backbone)'] = {'total': 5_000_000_000, 'trainable': None}
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    results['STAR (VSR modules)'] = None

# ─────────────────────────────────────────────────────────────
# 汇总表格
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("参数规模汇总表")
print("=" * 60)
print(f"{'模型':<30} {'总参数':>12} {'可训练参数':>14}")
print("-" * 60)
for name, res in results.items():
    if res is None:
        print(f"{name:<30} {'ERROR':>12} {'ERROR':>14}")
    elif res['trainable'] is None:
        print(f"{name:<30} {fmt(res['total']):>12} {'(仅推理)':>14}")
    else:
        print(f"{name:<30} {fmt(res['total']):>12} {fmt(res['trainable']):>14}")
print("=" * 60)
