"""
PSNR vs Runtime 气泡图（气泡大小 = 参数量）
- x 轴: 完整推理 Runtime (s, 100 帧)
- y 轴: REDS4 PSNR(Y) (dB)
- 气泡面积 ∝ 参数量 (M)

扩散模型 runtime 跨几个数量级，默认使用 log 轴。
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────
# 数据：REDS4 PSNR + 100帧完整推理 Runtime + 总参数量
# ─────────────────────────────────────────
MODELS = [
    # name,          PSNR,   runtime_s, trainable_M, is_ours
    ('ST-VSR',       27.61,   17.30,       2.95,  True),
    ('RealViformer', 28.17,    4.62,       5.82,  False),
    ('DiffVSR',      26.05, 1310.40,     949.57,  False),
    ('SCST',         25.79, 1082.25,     367.18,  False),
    ('STAR',         24.01,   96.68,    2041.12,  False),
]

USE_LOG_X = True     # runtime 跨 0.17s–2000s，需 log 轴
# 气泡半径按 log(params) 缩放，压缩 5M~3000M 的视觉差异
BUBBLE_RADIUS = 7.0    # 基础半径（points）
BUBBLE_GAIN   = 10.0   # 每 10× 参数量半径增量

# 各模型标签相对圆心的偏移 (dx, dy)，单位 points，便于手动微调
LABEL_OFFSETS = {
    'ST-VSR':       (0,  25),
    'RealViformer': (5,  25),
    'DiffVSR':      (0,  45),
    'SCST':         (0, -45),
    'STAR':         (0,  45),
}


def main():
    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)

    def _bubble_size(params_m):
        # 半径 = BUBBLE_RADIUS + BUBBLE_GAIN × log10(params)，面积 = π r²（points²）
        r = BUBBLE_RADIUS + BUBBLE_GAIN * np.log10(max(params_m, 1.0))
        return np.pi * r ** 2

    for name, psnr, rt, params, is_ours in MODELS:
        size = _bubble_size(params)
        color = '#FFB347' if is_ours else '#B9D7E8'   # Ours 橙色，其余浅蓝
        edge  = '#D97706' if is_ours else '#4A8BB8'
        ax.scatter(rt, psnr, s=size, c=color, edgecolors=edge,
                   linewidths=1.8, alpha=0.75, zorder=2)
        # 圆心黑点
        ax.scatter(rt, psnr, s=8, c='black', zorder=4)
        # 标签放在气泡外部，按 LABEL_OFFSETS 配置
        dx, dy = LABEL_OFFSETS.get(name, (0, 0))
        label = f"{name}\n(Ours)" if is_ours else name
        ax.annotate(label, (rt, psnr),
                    xytext=(dx, dy), textcoords='offset points',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold' if is_ours else 'normal',
                    zorder=3)

    # ── 轴设置 ──
    if USE_LOG_X:
        ax.set_xscale('log')
        ax.set_xlim(2.5, 6000)   # 右侧留空间避免 ~2000s 的气泡被裁切
    ax.set_xlabel('Runtime (s)  — 100 frames full inference', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.set_axisbelow(True)

    # y 轴范围
    psnrs = [m[1] for m in MODELS]
    ax.set_ylim(min(psnrs) - 1.0, max(psnrs) + 1.0)

    # ── 图例（手动绘制：气泡 + 内部文字 + 包裹框） ──
    import matplotlib.patches as mpatches
    legend_vals = [10, 100, 1000, 3000]   # M

    # 包裹框（axes 归一化坐标）
    frame_x0, frame_y0 = 0.3325, 0.70
    frame_x1, frame_y1 = 0.995, 0.985
    frame = mpatches.FancyBboxPatch(
        (frame_x0, frame_y0), frame_x1 - frame_x0, frame_y1 - frame_y0,
        boxstyle='round,pad=0.0,rounding_size=0.012',
        linewidth=1.0, edgecolor='gray', facecolor='white', alpha=0.9,
        transform=ax.transAxes, zorder=4)
    ax.add_patch(frame)

    # 标题 "Params"
    ax.text((frame_x0 + frame_x1) / 2, frame_y1 - 0.025,
            'Trainable Params', ha='center', va='top', fontsize=11,
            transform=ax.transAxes, zorder=5)

    ax.set_title('REDS4 — PSNR vs Runtime vs Trainable Params', fontsize=13, pad=10)
    plt.tight_layout()

    # 气泡及内部标签：气泡边缘之间等距
    # 需要在 tight_layout 之后计算 axes 像素宽度，才能把 points 半径换算到 axes 归一化坐标
    fig.canvas.draw()
    ax_w_px = ax.get_window_extent().width
    pad_in = 0.04  # 左右留边（axes 归一化）

    radii_axes = []
    for val in legend_vals:
        s = _bubble_size(val)                      # points²
        # scatter marker 的可见直径 = sqrt(s) points（而非 2·sqrt(s/π)）
        r_pts = np.sqrt(s) / 2.0
        r_ax  = r_pts * (fig.dpi / 72) / ax_w_px   # axes 归一化
        radii_axes.append(r_ax)

    inner_x0 = frame_x0 + pad_in
    inner_x1 = frame_x1 - pad_in
    available = inner_x1 - inner_x0
    total_diam = 2 * sum(radii_axes)
    n = len(legend_vals)
    gap = (available - total_diam) / (n - 1) if n > 1 else 0

    y_bubble = frame_y0 + (frame_y1 - frame_y0) * 0.38
    cur = inner_x0
    for val, r in zip(legend_vals, radii_axes):
        x = cur + r
        size = _bubble_size(val)
        ax.scatter(x, y_bubble, s=size, c='lightgray', edgecolors='gray',
                   linewidths=1.0, alpha=0.7,
                   transform=ax.transAxes, zorder=5, clip_on=False)
        ax.text(x, y_bubble, f'{val}M', ha='center', va='center',
                fontsize=9, color='#333',
                transform=ax.transAxes, zorder=6)
        cur += 2 * r + gap

    out_path = Path(__file__).resolve().parents[2] / 'outputs' / 'figures' / 'psnr_runtime_bubble.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
