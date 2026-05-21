"""
统一 Runtime Benchmark 脚本
对比以下 5 套模型在相同条件下的推理耗时：
  Ours / RealViformer / SCST / STAR / DiffVSR

运行环境说明：
  - Ours / RealViformer  : 当前 ST_VSR_Project venv（Python 3.13）直接运行
  - SCST / STAR / DiffVSR: 各自 venv 子进程运行，无需修改当前环境
    SCST   → /home/ubuntu/lib/SCST/.venv      (Python 3.10)
    STAR   → /home/ubuntu/lib/STAR/.venv      (Python 3.10)
    DiffVSR→ /home/ubuntu/lib/DiffVSR/.venv   (Python 3.9)

模型类型说明：
  - Ours / RealViformer 为单步推理（1 次前向 = 1 次完整推理）
  - SCST / STAR / DiffVSR 为扩散模型：
      * 脚本测量 1 个去噪步骤（UNet 单次前向）耗时
      * 完整推理 = N_steps × 单步耗时（典型值 20–50 步）

用法：
  python benchmark_runtime_unified.py
  python benchmark_runtime_unified.py --height 180 --width 320 --num_frames 100
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────
# 全局测速参数
# ──────────────────────────────────────────────────────────────
SCALE              = 4
DEFAULT_HEIGHT     = 180   # REDS4 LR 高度（720 ÷ 4）
DEFAULT_WIDTH      = 320   # REDS4 LR 宽度（1280 ÷ 4）
DEFAULT_NUM_FRAMES = 100   # REDS4 视频帧数
DEFAULT_WARMUP     = 10
DEFAULT_ITERATIONS = 30

# 扩散模型默认去噪步数（来自各自 pipeline __call__ 签名）
SCST_N_STEPS    = 20   # infer_mococtrl_REDS4.py line 39: --num_inference_steps 20 (评估脚本显式设定)
STAR_N_STEPS    = 15   # inference_sr.py: --steps 15 (fast solver, 脚本默认值)
DIFFVSR_N_STEPS = 50   # inference_tile.py line 290: --inference_steps default=50 (覆盖 pipeline 内部默认 75)
# ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description='Unified Runtime Benchmark')
    p.add_argument('--height',     type=int, default=DEFAULT_HEIGHT)
    p.add_argument('--width',      type=int, default=DEFAULT_WIDTH)
    p.add_argument('--num_frames', type=int, default=DEFAULT_NUM_FRAMES)
    p.add_argument('--warmup',     type=int, default=DEFAULT_WARMUP)
    p.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS)
    return p.parse_args()


def pad_to_multiple(h, w, m=8):
    ph = 0 if h % m == 0 else m - h % m
    pw = 0 if w % m == 0 else m - w % m
    return h + ph, w + pw


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def add_v100_hooks(model):
    """对所有 nn.Module 注册 pre-hook，强制输入张量连续（V100 兼容）"""
    def _hook(_m, args):
        return tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
    for m in model.modules():
        m.register_forward_pre_hook(_hook)


def patch_matmul_for_v100():
    """
    V100 non-contiguous CUBLAS 修复：同时 patch torch.matmul 函数和 Tensor.__matmul__。
    V100 的 cublasSgemmStridedBatched 不接受 non-contiguous 张量，
    而 q @ k.transpose(-2,-1) 产生的 transposed view 是 non-contiguous 的。
    """
    # patch torch.matmul（@ 在 C++ 层最终调用这里）
    _orig_matmul = torch.matmul
    def _safe_matmul(input, other, *args, **kwargs):
        return _orig_matmul(input.contiguous(), other.contiguous(), *args, **kwargs)
    torch.matmul = _safe_matmul

    # patch Tensor.__matmul__（Python 层 @ 运算符）
    _orig_mm = torch.Tensor.__matmul__
    def _safe_mm(self, other):
        return _orig_mm(self.contiguous(), other.contiguous())
    torch.Tensor.__matmul__ = _safe_mm

    # patch torch.bmm（部分实现走此路径）
    _orig_bmm = torch.bmm
    def _safe_bmm(input, mat2, *args, **kwargs):
        return _orig_bmm(input.contiguous(), mat2.contiguous(), *args, **kwargs)
    torch.bmm = _safe_bmm


def time_model(forward_fn, x, warmup, iterations, device):
    """CUDA Events 精确计时"""
    with torch.no_grad():
        for _ in range(warmup):
            forward_fn(x)
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    with torch.no_grad():
        for _ in range(iterations):
            forward_fn(x)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / 1000.0  # seconds


# ══════════════════════════════════════════════════════════════
# 本进程模型（Ours / RealViformer）
# ══════════════════════════════════════════════════════════════

def build_ours(device, H_lr, W_lr, H_hr, W_hr, T):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from models.st_network import ST_VSR_Network
    except ImportError as e:
        print(f"  [跳过 Ours] 导入失败: {e}")
        return None

    model = ST_VSR_Network(
        use_time_cond=True, use_shallow_cnn=True, use_semantic_prior=True
    ).eval().float().to(device)
    add_v100_hooks(model)

    y_c = (torch.arange(H_hr, dtype=torch.float32) + 0.5) / H_hr * 2.0 - 1.0
    x_c = (torch.arange(W_hr, dtype=torch.float32) + 0.5) / W_hr * 2.0 - 1.0
    gy, gx = torch.meshgrid(y_c, x_c, indexing='ij')
    cxy = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    ct  = torch.full((cxy.shape[0], 1), 0.0)
    coords = torch.cat([cxy, ct], dim=-1).unsqueeze(0).to(device)

    def forward_fn(x):
        # Ours 固定接受 3 帧（前帧/当前帧/后帧），从输入序列中取前 3 帧
        x3 = x[:, :3]  # (1, 3, 3, H, W)
        return model(x3, coords, chunk_size=30000)

    return 'Ours', forward_fn, count_params(model)


def _mdta_mm(q, k, v):
    """
    V100 workaround：PyTorch 在该版本对所有批量矩阵乘均调用 cublasSgemmStridedBatched，
    而 V100 对某些参数组合报 CUBLAS_STATUS_INVALID_VALUE。
    改用 torch.mm 循环（cublasSgemm，非批量）绕过此 bug。
    q/k/v shape: (B, H, C, HW)，返回 attn (B,H,C,C) 和 out (B,H,C,HW)。
    """
    B, H, C, HW = q.shape
    BH = B * H
    q_f = q.reshape(BH, C, HW)
    k_f = k.reshape(BH, C, HW)
    v_f = v.reshape(BH, C, HW)
    attn_list = [torch.mm(q_f[i], k_f[i].t()) for i in range(BH)]
    attn_flat = torch.stack(attn_list)           # (BH, C, C)
    out_list  = [torch.mm(attn_flat[i], v_f[i]) for i in range(BH)]
    return attn_flat.reshape(B, H, C, C), torch.stack(out_list).reshape(B, H, C, HW)


def _patch_realviformer_v100():
    """
    V100 CUBLAS 修复：patch RealViformer 所有 Attention 类，用 torch.mm 循环替代批量矩阵乘。
    """
    import torch.nn.functional as _F
    from einops import rearrange as _re
    import torch as _t

    # ── 1. realviformer_arch.Attention ──
    from archs.realviformer_arch import Attention as _RvAttn
    def _rv_attn_fwd(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = _re(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = _re(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = _re(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = _F.normalize(q, dim=-1); k = _F.normalize(k, dim=-1)
        attn, out = _mdta_mm(q, k, v)
        attn = attn * self.temperature
        if self.withmask:
            b_, h_, c_, _ = attn.shape
            mi = attn.clone()
            dsc = _t.cat([self.max_dsc(mi.reshape(b_, h_*c_, c_)),
                          self.avg_dsc(mi.reshape(b_, h_*c_, c_))], dim=-1).reshape(b_, h_, c_, 2)
            mask = _F.sigmoid(self.linear2(_F.gelu(self.linear1(dsc)).transpose(-2,-1)).transpose(-2,-1))
        attn = attn.softmax(dim=-1)
        # re-compute out with softmax'd attn
        BH = b * self.num_heads
        out_list = [_t.mm(attn.reshape(BH, c//self.num_heads, c//self.num_heads)[i],
                          v.reshape(BH, c//self.num_heads, h*w)[i]) for i in range(BH)]
        out = _t.stack(out_list).reshape(b, self.num_heads, c//self.num_heads, h*w)
        if self.withmask:
            out = out * mask
        out = _re(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)
    _RvAttn.forward = _rv_attn_fwd

    # ── 2. arch_util.Attention ──
    from archs.arch_util import Attention as _AuAttn
    def _au_attn_fwd(self, x, y):
        b, c_kv, h, w = x.shape
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q(y)
        q = _re(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = _re(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = _re(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = _F.normalize(q, dim=-1); k = _F.normalize(k, dim=-1)
        attn, _ = _mdta_mm(q, k, v)
        attn = (attn * self.temperature).softmax(dim=-1)
        BH = b * self.num_heads; Ck = k.shape[2]; HW = h*w
        out_list = [_t.mm(attn.reshape(BH, q.shape[2], Ck)[i], v.reshape(BH, Ck, HW)[i]) for i in range(BH)]
        out = _re(_t.stack(out_list).reshape(b, self.num_heads, q.shape[2], HW),
                  'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)
    _AuAttn.forward = _au_attn_fwd

    # ── 3. arch_util.CrossChannelAttention ──
    from archs.arch_util import CrossChannelAttention as _CCA
    def _cca_fwd(self, x, y, return_attn=False):
        b, c_kv, h, w = x.shape
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q(x)
        q = _re(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = _re(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = _re(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = _F.normalize(q, dim=-1); k = _F.normalize(k, dim=-1)
        BH = b * self.num_heads; Cq = q.shape[2]; Ck = k.shape[2]; HW = h*w
        q_f, k_f, v_f = q.reshape(BH, Cq, HW), k.reshape(BH, Ck, HW), v.reshape(BH, Ck, HW)
        attn_list = [_t.mm(q_f[i], k_f[i].t()) for i in range(BH)]
        attn_flat = (_t.stack(attn_list).reshape(b, self.num_heads, Cq, Ck) * self.temperature).softmax(dim=-1)
        out_list = [_t.mm(attn_flat.reshape(BH, Cq, Ck)[i], v_f[i]) for i in range(BH)]
        out = _re(_t.stack(out_list).reshape(b, self.num_heads, Cq, HW),
                  'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return (out, attn_flat) if return_attn else out
    _CCA.forward = _cca_fwd

    # ── 4. arch_util.cross_attention（模块级函数）──
    import archs.arch_util as _au_mod
    def _ca_fn(x, ref):
        b, c0, h, w = x.shape
        q, k, v = x.clone(), ref.clone(), ref.clone()
        q = _re(q, 'b (head c) h w -> b head c (h w)', head=1)
        k = _re(k, 'b (head c) h w -> b head c (h w)', head=1)
        v = _re(v, 'b (head c) h w -> b head c (h w)', head=1)
        q = _F.normalize(q, dim=-1); k = _F.normalize(k, dim=-1)
        HW = h * w; C0 = q.shape[2]
        attn = _t.mm(q.reshape(C0, HW), k.reshape(C0, HW).t())
        max_, _ = _t.max(attn, dim=-1, keepdim=True)
        attn = (attn.softmax(dim=-1) * _t.lt(attn - max_, 0).int()).softmax(dim=-1)
        out = _t.mm(attn, v.reshape(C0, HW))
        return _re(out.reshape(b, 1, C0, HW), 'b head c (h w) -> b (head c) h w', head=1, h=h, w=w)
    _au_mod.cross_attention = _ca_fn


def build_realviformer(device):
    sys.path.insert(0, '/home/ubuntu/lib/RealViformer')
    try:
        from archs.realviformer_arch import RealViformer
    except ImportError as e:
        print(f"  [跳过 RealViformer] 导入失败: {e}")
        return None

    _patch_realviformer_v100()   # 必须在模型实例化之前或之后均可（patch 类方法）

    model = RealViformer(
        num_feat=48, num_blocks=[2, 3, 4, 1], spynet_path=None,
        heads=[1, 2, 4], ffn_expansion_factor=2.66, merge_head=2,
        bias=False, LayerNorm_type='BiasFree',
        ch_compress=True, squeeze_factor=[4, 4, 4], masked=True,
    ).eval().to(device)
    add_v100_hooks(model)

    def forward_fn(x):
        return model(x)

    return 'RealViformer', forward_fn, count_params(model)


# ══════════════════════════════════════════════════════════════
# 子进程模型（SCST / STAR / DiffVSR）
# ══════════════════════════════════════════════════════════════

_V100_PATCH = """
import torch
# patch torch.matmul
_omm = torch.matmul
def _smm(i, o, *a, **k): return _omm(i.contiguous(), o.contiguous(), *a, **k)
torch.matmul = _smm
# patch Tensor.__matmul__
_otmm = torch.Tensor.__matmul__
torch.Tensor.__matmul__ = lambda s, o: _otmm(s.contiguous(), o.contiguous())
# patch torch.bmm
_obmm = torch.bmm
torch.bmm = lambda i, m, *a, **k: _obmm(i.contiguous(), m.contiguous(), *a, **k)
def _add_hooks(model):
    def _h(_m, a): return tuple(x.contiguous() if isinstance(x, torch.Tensor) else x for x in a)
    for m in model.modules(): m.register_forward_pre_hook(_h)
"""

_TIMING_CODE = """
import torch
device = torch.device('cuda')
def _time(forward_fn, warmup, iterations):
    with torch.no_grad():
        for _ in range(warmup): forward_fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    with torch.no_grad():
        for _ in range(iterations): forward_fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / 1000.0
"""


def _make_ours_script(H_lr, W_lr, warmup, iterations):
    """Ours 子进程脚本：3帧输入→1帧输出，fps = 1/per_inference_time"""
    H_hr = H_lr * SCALE
    W_hr = W_lr * SCALE
    return f"""
import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r})
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
{_V100_PATCH}
{_TIMING_CODE}
from models.st_network import ST_VSR_Network
model = ST_VSR_Network(
    use_time_cond=True, use_shallow_cnn=True, use_semantic_prior=True
).eval().float().to(device)
_add_hooks(model)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

y_c = (torch.arange({H_hr}, dtype=torch.float32) + 0.5) / {H_hr} * 2.0 - 1.0
x_c = (torch.arange({W_hr}, dtype=torch.float32) + 0.5) / {W_hr} * 2.0 - 1.0
gy, gx = torch.meshgrid(y_c, x_c, indexing='ij')
cxy = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
ct  = torch.full((cxy.shape[0], 1), 0.0)
coords = torch.cat([cxy, ct], dim=-1).unsqueeze(0).to(device)

x = torch.rand(1, 3, 3, {H_lr}, {W_lr}, device=device)  # 3帧输入
def fwd():
    return model(x, coords, chunk_size=30000)
total_s = _time(fwd, {warmup}, {iterations})
per_ms = total_s / {iterations} * 1000
fps = 1.0 / (total_s / {iterations})   # 1 输出帧/推理
print(f"RESULT:{{n_params:.4f}}:{{n_trainable:.4f}}:{{per_ms:.4f}}:{{fps:.4f}}")
"""


def _make_realviformer_script(H_lr, W_lr, T, warmup, iterations):
    """RealViformer 子进程脚本：T帧输入→T帧输出，含V100 CUBLAS patch"""
    return f"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '/home/ubuntu/lib/RealViformer')
{_V100_PATCH}
{_TIMING_CODE}
import torch as _t
import torch.nn.functional as _F
from einops import rearrange as _re

def _mdta_mm(q, k, v):
    B, H, C, HW = q.shape; BH = B * H
    qf, kf, vf = q.reshape(BH,C,HW), k.reshape(BH,C,HW), v.reshape(BH,C,HW)
    afl = [_t.mm(qf[i], kf[i].t()) for i in range(BH)]
    af  = _t.stack(afl)
    ofl = [_t.mm(af[i], vf[i]) for i in range(BH)]
    return af.reshape(B,H,C,C), _t.stack(ofl).reshape(B,H,C,HW)

# ── 1. realviformer_arch.Attention ──
from archs.realviformer_arch import Attention as _RvAttn
def _rv_fwd(self, x):
    b,c,h,w = x.shape
    qkv = self.qkv_dwconv(self.qkv(x))
    q,k,v = qkv.chunk(3, dim=1)
    q = _re(q,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    k = _re(k,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    v = _re(v,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    q = _F.normalize(q,dim=-1); k = _F.normalize(k,dim=-1)
    attn, out = _mdta_mm(q, k, v)
    attn = attn * self.temperature
    if self.withmask:
        b_,h_,c_,_ = attn.shape
        mi = attn.clone()
        dsc = _t.cat([self.max_dsc(mi.reshape(b_,h_*c_,c_)),
                      self.avg_dsc(mi.reshape(b_,h_*c_,c_))],dim=-1).reshape(b_,h_,c_,2)
        mask = _F.sigmoid(self.linear2(_F.gelu(self.linear1(dsc)).transpose(-2,-1)).transpose(-2,-1))
    attn = attn.softmax(dim=-1)
    BH = b*self.num_heads; Ck = k.shape[2]; HW = h*w
    ol = [_t.mm(attn.reshape(BH,c//self.num_heads,c//self.num_heads)[i],
                v.reshape(BH,c//self.num_heads,h*w)[i]) for i in range(BH)]
    out = _t.stack(ol).reshape(b,self.num_heads,c//self.num_heads,h*w)
    if self.withmask: out = out * mask
    out = _re(out,'b head c (h w) -> b (head c) h w',head=self.num_heads,h=h,w=w)
    return self.project_out(out)
_RvAttn.forward = _rv_fwd

# ── 2. arch_util.Attention ──
from archs.arch_util import Attention as _AuAttn
def _au_fwd(self, x, y):
    b,c_kv,h,w = x.shape
    kv = self.kv_dwconv(self.kv(x)); k,v = kv.chunk(2,dim=1)
    q = self.q(y)
    q = _re(q,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    k = _re(k,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    v = _re(v,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    q = _F.normalize(q,dim=-1); k = _F.normalize(k,dim=-1)
    attn,_ = _mdta_mm(q,k,v)
    attn = (attn*self.temperature).softmax(dim=-1)
    BH=b*self.num_heads; Ck=k.shape[2]; HW=h*w
    ol = [_t.mm(attn.reshape(BH,q.shape[2],Ck)[i],v.reshape(BH,Ck,HW)[i]) for i in range(BH)]
    out = _re(_t.stack(ol).reshape(b,self.num_heads,q.shape[2],HW),
              'b head c (h w) -> b (head c) h w',head=self.num_heads,h=h,w=w)
    return self.project_out(out)
_AuAttn.forward = _au_fwd

# ── 3. arch_util.CrossChannelAttention ──
from archs.arch_util import CrossChannelAttention as _CCA
def _cca_fwd(self, x, y, return_attn=False):
    b,c_kv,h,w = x.shape
    kv = self.kv_dwconv(self.kv(y)); k,v = kv.chunk(2,dim=1)
    q = self.q(x)
    q=_re(q,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    k=_re(k,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    v=_re(v,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    q=_F.normalize(q,dim=-1); k=_F.normalize(k,dim=-1)
    BH=b*self.num_heads; Cq=q.shape[2]; Ck=k.shape[2]; HW=h*w
    qf,kf,vf=q.reshape(BH,Cq,HW),k.reshape(BH,Ck,HW),v.reshape(BH,Ck,HW)
    al=[_t.mm(qf[i],kf[i].t()) for i in range(BH)]
    af=(_t.stack(al).reshape(b,self.num_heads,Cq,Ck)*self.temperature).softmax(dim=-1)
    ol=[_t.mm(af.reshape(BH,Cq,Ck)[i],vf[i]) for i in range(BH)]
    out=_re(_t.stack(ol).reshape(b,self.num_heads,Cq,HW),
            'b head c (h w) -> b (head c) h w',head=self.num_heads,h=h,w=w)
    out=self.project_out(out)
    return (out,af) if return_attn else out
_CCA.forward = _cca_fwd

# ── 4. arch_util.cross_attention ──
import archs.arch_util as _au_mod
def _ca_fn(x, ref):
    b,c0,h,w=x.shape
    q=_re(x.clone(),'b (head c) h w -> b head c (h w)',head=1)
    k=_re(ref.clone(),'b (head c) h w -> b head c (h w)',head=1)
    v=_re(ref.clone(),'b (head c) h w -> b head c (h w)',head=1)
    q=_F.normalize(q,dim=-1); k=_F.normalize(k,dim=-1)
    HW=h*w; C0=q.shape[2]
    attn=_t.mm(q.reshape(C0,HW),k.reshape(C0,HW).t())
    mx,_=_t.max(attn,dim=-1,keepdim=True)
    attn=(attn.softmax(dim=-1)*_t.lt(attn-mx,0).int()).softmax(dim=-1)
    out=_t.mm(attn,v.reshape(C0,HW))
    return _re(out.reshape(b,1,C0,HW),'b head c (h w) -> b (head c) h w',head=1,h=h,w=w)
_au_mod.cross_attention = _ca_fn

from archs.realviformer_arch import RealViformer
model = RealViformer(
    num_feat=48, num_blocks=[2,3,4,1], spynet_path=None,
    heads=[1,2,4], ffn_expansion_factor=2.66, merge_head=2,
    bias=False, LayerNorm_type='BiasFree',
    ch_compress=True, squeeze_factor=[4,4,4], masked=True,
).eval().to(device)
_add_hooks(model)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
x = torch.rand(1, {T}, 3, {H_lr}, {W_lr}, device=device)
def fwd():
    return model(x)
total_s = _time(fwd, {warmup}, {iterations})
per_ms = total_s / {iterations} * 1000
fps = {T} / (total_s / {iterations})
print(f"RESULT:{{n_params:.4f}}:{{n_trainable:.4f}}:{{per_ms:.4f}}:{{fps:.4f}}")
"""


def _make_scst_script(H_lr, W_lr, T, warmup, iterations):
    latent_H = (H_lr * SCALE) // 8
    latent_W = (W_lr * SCALE) // 8
    # SCST 使用滑动窗口推理：pipeline 默认 num_frame=5，每次 UNet call 只处理 5 帧
    # 不能用 T=100，否则中间激活 ×20 直接 OOM
    T_scst = 5          # 与 pipeline_SCST.py 默认 num_frame 保持一致
    chunks = (T + T_scst - 1) // T_scst   # 处理 T 帧所需的 UNet call 次数
    return f"""
import sys; sys.path.insert(0, '.')
{_V100_PATCH}
{_TIMING_CODE}
# SCST pipeline 以滑动窗口处理视频（默认 num_frame=5 帧/次）
# 处理 {T} 帧需 {chunks} 次（ControlNet+UNet）call；fps = {T} / (单窗口耗时 × {chunks})
# 每步真实耗时 = ControlNet forward + UNet forward（两者均纳入计时）
from models.vsr.unet_3d import UNet3DConditionModel
from models.controlnet.controlnet import ControlNetModel
from einops import rearrange

# 实例化 ControlNet（随机权重，仅用于推断残差形状）
controlnet = ControlNetModel(
    in_channels=4,
    block_out_channels=(320, 640, 1280, 1280),
    layers_per_block=2,
    cross_attention_dim=1280,
    attention_head_dim=8,
).eval().float().to(device)
_add_hooks(controlnet)

# 实例化 UNet
model = UNet3DConditionModel(
    in_channels=4, out_channels=4,
    block_out_channels=(320, 640, 1280, 1280),
    layers_per_block=2, cross_attention_dim=1280,
    attention_head_dim=8, use_motion_module=False,
).eval().float().to(device)
_add_hooks(model)

# 完整 pipeline 参数统计（含 VAE / CLIP / ControlNet / UNet）
# CLIPTextModel (SD1.5 config, 无需下载预训练权重)
from transformers import CLIPTextConfig, CLIPTextModel as _CLIP
_clip_cfg = _CLIP.config_class(vocab_size=49408, hidden_size=768, intermediate_size=3072,
    num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=77,
    hidden_act='quick_gelu', layer_norm_eps=1e-5)
_clip = _CLIP(_clip_cfg)
n_clip = sum(p.numel() for p in _clip.parameters()) / 1e6
del _clip
# AutoencoderKL (SD1.5 VAE config, 无需下载)
from diffusers.models import AutoencoderKL as _AKL
_vae = _AKL(in_channels=3, out_channels=3,
    down_block_types=('DownEncoderBlock2D',)*4,
    up_block_types=('UpDecoderBlock2D',)*4,
    block_out_channels=(128,256,512,512), layers_per_block=2, latent_channels=4)
n_vae = sum(p.numel() for p in _vae.parameters()) / 1e6
del _vae
n_controlnet = sum(p.numel() for p in controlnet.parameters()) / 1e6
n_unet_only  = sum(p.numel() for p in model.parameters()) / 1e6
n_params = n_clip + n_vae + n_controlnet + n_unet_only  # 全 pipeline 参数
n_trainable = n_controlnet  # SCST 训练时仅 ControlNet 可训练，CLIP/VAE/UNet 冻结

# 真实每步推理 = ControlNet forward + UNet forward（两者均需计时）
# controlnet_cond 是 pixel-space 图像（3通道RGB），不是 latent（4通道）
H_pixel = {latent_H} * 8  # 恢复到像素空间分辨率
W_pixel = {latent_W} * 8
dummy_unet_h   = torch.zeros(1, 77, 1280, device=device)    # UNet encoder_hidden_states
dummy_enc_hs   = torch.zeros({T_scst}, 77, 1280, device=device)  # ControlNet enc hidden
dummy_cond_img = torch.zeros({T_scst}, 3, H_pixel, W_pixel, device=device)  # ControlNet cond
dummy_ts       = torch.tensor([500]*{T_scst}, device=device)

def fwd():
    lat    = torch.rand(1, 4, {T_scst}, {latent_H}, {latent_W}, device=device)
    lat_2d = lat.squeeze(0).permute(1, 0, 2, 3)  # (T_scst, 4, latent_H, latent_W)
    # Step A: ControlNet forward（每步必须运行，产生随 latent 变化的残差）
    ctrl = controlnet(lat_2d, timestep=dummy_ts,
                      encoder_hidden_states=dummy_enc_hs,
                      controlnet_cond=dummy_cond_img, return_dict=False)
    down_res = [rearrange(r, "(b f) c h w -> b c f h w", b=1, f={T_scst})
                for r in ctrl[2][0]]
    mid_res  = rearrange(ctrl[3][0], "(b f) c h w -> b c f h w", b=1, f={T_scst})
    # Step B: UNet forward（接收 ControlNet 残差）
    return model(lat, timestep=500, encoder_hidden_states=dummy_unet_h,
                 down_block_additional_residuals=down_res,
                 mid_block_additional_residual=mid_res)
total_s = _time(fwd, {warmup}, {iterations})
per_ms_per_chunk = total_s / {iterations} * 1000   # 每 {T_scst} 帧窗口（ControlNet+UNet）的耗时
# 处理 {T} 帧需 {chunks} 次（ControlNet+UNet）call，fps 以整体 {T} 帧计算
fps = {T} / (total_s / {iterations} * {chunks})
print(f"RESULT:{{n_params:.4f}}:{{n_trainable:.4f}}:{{per_ms_per_chunk:.4f}}:{{fps:.4f}}")
"""


def _make_star_script(H_lr, W_lr, T, warmup, iterations):
    # STAR 在 LR 隐空间扩散（latent = LR/8），而非 HR 隐空间
    # STAR 的 Downsample 使用非对称 padding=(2,1)，Upsample 只裁剪 H 首尾各1行
    # 兼容条件：latent_H ≡ 2 (mod 8)，latent_W ≡ 0 (mod 8)
    # REDS4 LR 320×180 对应隐空间: W=320/8=40(≡0 mod8 ✓), H=180/8=22.5
    # STAR 架构约束: H≡2(mod8)且W≡0(mod8)，取最近合法值 H=26(=8×3+2)
    star_latent_H = 26   # ≡ 2 (mod 8)，对应像素 H ≈ 208
    star_latent_W = 40   # = 320/8，对应像素 W = 320
    # Pipeline (inference_sr.py) 默认 max_chunk_len=32：100帧分成5个chunk
    # make_chunks(100, interp_f_num=0, max_chunk_len=32, overlap_ratio=0.5):
    #   chunk_len=32, o_len=16, stride=16 → [(0,32),(16,48),(32,64),(48,80),(64,100)] = 5 chunks
    T_star = 32    # 每次 UNet call 处理 32 帧
    chunks = 5     # 处理 T=100 帧所需 chunk 数
    return f"""
import sys; sys.path.insert(0, '.')
{_V100_PATCH}
{_TIMING_CODE}
# STAR pipeline 以 max_chunk_len=32 的滑动窗口处理视频
# 处理 {T} 帧需 {chunks} 次 UNet call；fps = {T} / (单块耗时 × {chunks})
from video_to_video.modules.unet_v2v import ControlledV2VUNet
model = ControlledV2VUNet()
ckpt_path = '/home/ubuntu/data/STAR/pretrained_weight/heavy_deg.pt'
load_dict = torch.load(ckpt_path, map_location='cpu')
if 'state_dict' in load_dict:
    load_dict = load_dict['state_dict']
model.load_state_dict(load_dict, strict=False)
# float32：sinusoidal timestep embedding 产生 float32，half() 会引发 dtype 不匹配
model = model.float().to(device).eval()
_add_hooks(model)
# 完整 pipeline 参数统计：OpenCLIP ViT-H-14 + SVD VAE + ControlledV2VUNet
import open_clip as _oc
_clip = _oc.create_model('ViT-H-14')
n_clip = sum(p.numel() for p in _clip.parameters()) / 1e6
del _clip
from diffusers import AutoencoderKLTemporalDecoder as _SVD_VAE
_svd_vae = _SVD_VAE(in_channels=3, out_channels=3,
    down_block_types=('DownEncoderBlock2D','DownEncoderBlock2D','DownEncoderBlock2D'),
    block_out_channels=(128,256,512), layers_per_block=2, latent_channels=4)
n_vae = sum(p.numel() for p in _svd_vae.parameters()) / 1e6
del _svd_vae
n_unet_only = sum(p.numel() for p in model.parameters()) / 1e6
n_params = n_clip + n_vae + n_unet_only
n_trainable = n_unet_only  # STAR 仅 ControlledV2VUNet 可训练，CLIP/VAE 冻结
dummy_y    = torch.zeros(1, 77, 1024, device=device)
dummy_hint = torch.zeros(1, 4, {T_star}, {star_latent_H}, {star_latent_W}, device=device)
def fwd():
    lat = torch.rand(1, 4, {T_star}, {star_latent_H}, {star_latent_W}, device=device)
    t   = torch.tensor([500], dtype=torch.long, device=device)
    return model(lat, t, dummy_y, hint=dummy_hint)
total_s = _time(fwd, {warmup}, {iterations})
per_ms_per_chunk = total_s / {iterations} * 1000   # 每 {T_star} 帧块耗时
fps = {T} / (total_s / {iterations} * {chunks})     # 折算到处理 {T} 帧
print(f"RESULT:{{n_params:.4f}}:{{n_trainable:.4f}}:{{per_ms_per_chunk:.4f}}:{{fps:.4f}}")
"""


def _make_diffvsr_script(H_lr, W_lr, T, warmup, iterations):
    latent_H = (H_lr * SCALE) // 8
    latent_W = (W_lr * SCALE) // 8
    # Pipeline (pipeline_stable_diffusion_DiffVSR.py) 固定 window_size=8, stride=4
    # chunk_starts = range(0, T - window_size + stride, stride) = range(0, 96, 4) → 24 chunks for T=100
    T_diffvsr = 8    # window_size（每次 UNet call 处理 8 帧）
    stride_d  = 4    # 滑动步长
    chunks = len(list(range(0, T - T_diffvsr + stride_d, stride_d)))   # = 24 for T=100
    return f"""
import sys; sys.path.insert(0, '.')
{_V100_PATCH}
{_TIMING_CODE}
# DiffVSR pipeline 以 window_size=8, stride=4 的滑动窗口处理视频
# 处理 {T} 帧需 {chunks} 次 UNet call；fps = {T} / (单窗口耗时 × {chunks})
from models.unet import UNet3DVSRModel
cfg   = UNet3DVSRModel.load_config('./configs/unet_3d_config.json')
model = UNet3DVSRModel.from_config(cfg).eval().float().to(device)
_add_hooks(model)
# 完整 pipeline 参数统计：CLIPTextModel + TE-3DVAE + UNet3DVSRModel
from transformers import CLIPTextConfig, CLIPTextModel as _CLIP
_clip_cfg = _CLIP.config_class(vocab_size=49408, hidden_size=768, intermediate_size=3072,
    num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=77,
    hidden_act='quick_gelu', layer_norm_eps=1e-5)
_clip = _CLIP(_clip_cfg)
n_clip = sum(p.numel() for p in _clip.parameters()) / 1e6
del _clip
from models.autoencoder_kl_TE_3DVAE import AutoencoderKLTemporalDecoder as _TE3DVAE
_vae_cfg = _TE3DVAE.load_config('./configs/vae_config.json')
_vae = _TE3DVAE.from_config(_vae_cfg)
n_vae = sum(p.numel() for p in _vae.parameters()) / 1e6
del _vae
n_unet_only = sum(p.numel() for p in model.parameters()) / 1e6
n_params = n_clip + n_vae + n_unet_only
n_trainable = n_unet_only  # DiffVSR 仅 UNet3DVSRModel 可训练，CLIP/VAE 冻结
# class_labels 必须为 Tensor；encoder_hidden_states 必须非 None（cross_attention_dim=1024）
cls    = torch.tensor([20], device=device)
enc_hs = torch.zeros(1, 77, 1024, device=device)
def fwd():
    sample  = torch.rand(1, 4, {T_diffvsr}, {latent_H}, {latent_W}, device=device)
    low_res = torch.rand(1, 3, {T_diffvsr}, {latent_H}, {latent_W}, device=device)
    return model(sample, timestep=500, low_res=low_res,
                 class_labels=cls, encoder_hidden_states=enc_hs)
total_s = _time(fwd, {warmup}, {iterations})
per_ms_per_chunk = total_s / {iterations} * 1000   # 每 {T_diffvsr} 帧窗口耗时
fps = {T} / (total_s / {iterations} * {chunks})     # 折算到处理 {T} 帧
print(f"RESULT:{{n_params:.4f}}:{{n_trainable:.4f}}:{{per_ms_per_chunk:.4f}}:{{fps:.4f}}")
"""


def run_subprocess(python_bin, cwd, script, model_name, timeout=300):
    """运行子进程 benchmark，解析 RESULT: 行，返回 (name, n_params, n_trainable, per_ms, fps) 或 None"""
    print(f"[{model_name}]  子进程测速中...")
    try:
        result = subprocess.run(
            [python_bin, '-c', script],
            cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            print(f"  [错误] 子进程退出码 {result.returncode}")
            for line in result.stderr.splitlines()[-8:]:
                print(f"    {line}")
            return None, None, None, None, None

        for line in result.stdout.splitlines():
            if line.startswith('RESULT:'):
                parts = line.split(':')
                n_params    = float(parts[1])
                n_trainable = float(parts[2])
                per_ms      = float(parts[3])
                fps         = float(parts[4])
                print(f"  单次推理: {per_ms:.2f} ms  |  {fps:.2f} FPS (等效帧率)  |  可训练: {n_trainable:.2f}M")
                return model_name, n_params, n_trainable, per_ms, fps

        print(f"  [错误] 未找到 RESULT 行，stdout:")
        print(result.stdout[-400:])
        return None, None, None, None, None

    except subprocess.TimeoutExpired:
        print(f"  [超时] 超过 {timeout}s")
        return None, None, None, None, None
    except Exception as e:
        print(f"  [异常] {e}")
        return None, None, None, None, None


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"设备: {torch.cuda.get_device_name(device)}\n")
    else:
        raise RuntimeError("未检测到 GPU，benchmark 无意义")

    H, W = pad_to_multiple(args.height, args.width, 8)
    H_hr, W_hr = H * SCALE, W * SCALE
    T = args.num_frames
    WM, IT = args.warmup, args.iterations

    print("统一测速条件:")
    print(f"  LR 分辨率  : {H} × {W}")
    print(f"  HR 分辨率  : {H_hr} × {W_hr}  (×{SCALE} SR)")
    print(f"  序列帧数   : {T}")
    print(f"  预热次数   : {WM}")
    print(f"  计时迭代   : {IT}")
    print()

    # 所有模型均以独立子进程运行，确保各自 CUDA 上下文完全隔离，
    # 避免模型间显存泄漏/缓存污染。主进程不分配任何 GPU 显存。
    results = []
    THIS_PYTHON = sys.executable       # ST_VSR_Project venv，Python 3.13
    THIS_CWD    = str(PROJECT_ROOT)

    # ── 1. Ours (子进程) ──
    # 固定 3 帧输入（前帧/当前帧/后帧）→ 1 帧输出，fps = 1/推理时间
    ours_script = _make_ours_script(H, W, WM, IT)
    name, n_params, n_trainable, per_ms, fps = run_subprocess(
        THIS_PYTHON, THIS_CWD, ours_script, 'Ours', timeout=600
    )
    results.append((name or 'Ours', n_params or float('nan'), n_trainable or float('nan'), per_ms, fps, 1))

    # ── 2. RealViformer (子进程) ──
    rv_script = _make_realviformer_script(H, W, T, WM, IT)
    name, n_params, n_trainable, per_ms, fps = run_subprocess(
        THIS_PYTHON, THIS_CWD, rv_script, 'RealViformer', timeout=600
    )
    results.append((name or 'RealViformer', n_params or float('nan'), n_trainable or float('nan'), per_ms, fps, 1))

    # ── 3. SCST (子进程) ──
    # SCST 的 UNet 强依赖 ControlNet 输出（down_block_additional_residuals 不可为 None）
    # 修复：在子进程中同时实例化 ControlNet，先运行一次得到正确形状的残差，再计时 UNet
    scst_script = _make_scst_script(H, W, T, WM, IT)
    name, n_params, n_trainable, per_ms, fps = run_subprocess(
        '/home/ubuntu/lib/SCST/.venv/bin/python',
        '/home/ubuntu/lib/SCST',
        scst_script, 'SCST', timeout=600
    )
    results.append(('SCST', n_params or float('nan'), n_trainable or float('nan'), per_ms, fps, SCST_N_STEPS))

    # ── 4. STAR (子进程) ──
    # STAR 需要加载预训练权重：随机初始化时 VideoControlNet skip 尺寸与主 UNet 不兼容
    star_script = _make_star_script(H, W, T, WM, IT)
    name, n_params, n_trainable, per_ms, fps = run_subprocess(
        '/home/ubuntu/lib/STAR/.venv/bin/python',
        '/home/ubuntu/lib/STAR',
        star_script, 'STAR', timeout=600
    )
    results.append(('STAR', n_params or float('nan'), n_trainable or float('nan'), per_ms, fps, STAR_N_STEPS))

    # ── 5. DiffVSR (子进程) ──
    diffvsr_script = _make_diffvsr_script(H, W, T, WM, IT)
    name, n_params, n_trainable, per_ms, fps = run_subprocess(
        '/home/ubuntu/lib/DiffVSR/.venv/bin/python',
        '/home/ubuntu/lib/DiffVSR',
        diffvsr_script, 'DiffVSR', timeout=900
    )
    results.append(('DiffVSR', n_params or float('nan'), n_trainable or float('nan'), per_ms, fps, DIFFVSR_N_STEPS))

    # ── 汇总表格（完整推理） ──
    # 完整推理时间 = 单步耗时 × N_steps（对扩散模型）
    # 单步 fps 已按实际 sliding-window chunks 折算到处理 T 帧的等效帧率
    # → 完整推理 fps = 单步 fps / N_steps
    print()
    print("=" * 90)
    print(f"  {'模型':<16} {'参数(M)':>8} {'可训练(M)':>10} {'步数':>6} {'Runtime(ms)':>12} {'单步FPS':>10} {'完整推理FPS':>13}")
    print("-" * 90)
    for row in results:
        name, n_params, n_trainable, per_ms, fps_1step, n_steps = row
        if per_ms is None:
            print(f"  {name:<16} {n_params:>8.2f} {n_trainable:>10.2f} {n_steps:>6} {'ERROR':>12} {'ERROR':>10} {'ERROR':>13}")
        else:
            full_fps = fps_1step / n_steps
            step_label = f"{n_steps}步" if n_steps > 1 else "单步"
            print(f"  {name:<16} {n_params:>8.2f} {n_trainable:>10.2f} {step_label:>6} {per_ms:>12.2f} {fps_1step:>10.2f} {full_fps:>13.4f}")
    print("=" * 90)
    print()
    print("说明：")
    print("  - Ours / RealViformer：单次前向 = 完整推理，步数=1，单步FPS = 完整推理FPS。")
    print(f"  - SCST / STAR / DiffVSR：扩散模型，完整推理需多步去噪。")
    print(f"      SCST ({SCST_N_STEPS}步) ：(ControlNet+UNet) 5帧/次 × 20次/步 → 完整推理FPS = 单步FPS / {SCST_N_STEPS}  [infer_mococtrl_REDS4.py]")
    print(f"      STAR ({STAR_N_STEPS}步) ：UNet 32帧/次 × 5次/步  → 完整推理FPS = 单步FPS / {STAR_N_STEPS}  [inference_sr.py --steps 15]")
    print(f"      DiffVSR ({DIFFVSR_N_STEPS}步)：UNet 8帧/次 × 24次/步 → 完整推理FPS = 单步FPS / {DIFFVSR_N_STEPS}  [inference_tile.py --inference_steps 50]")
    print("  - 所有 FPS 均按处理 100 帧（REDS4 视频长度）折算。")


if __name__ == '__main__':
    main()
