#!/bin/bash
# 对 epoch_45/50/55/60/65/70 六组 EMA 模型在 Vid4、UDM10、REDS4 上进行评估
# 评估结果写入 results/metrics/eval_45_50_55_60_65_70.md

set -e

cd /home/ubuntu/lib/ST_VSR_Project

EPOCHS=(45 50 55 60 65 70)
EXP=Ours_DeformINR_LatentPrior_Ep55
OUT_MD="results/metrics/eval_45_50_55_60_65_70.md"

GT_VID4="/home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/GT"
GT_UDM10="/home/ubuntu/data/UDM10/GT"
GT_REDS4="/home/ubuntu/data/REDS4/train_sharp"

mkdir -p results/metrics

# ============================================================
# 写 Markdown 头部
# ============================================================
cat > "$OUT_MD" << 'EOF'
# DeformINR_LatentPrior — Epoch 45/50/55/60/65/70 评估报告

> 评估数据集：Vid4 / UDM10 / REDS4，超分倍率 4×，`--crop_border 4`
> 包含画质指标（PSNR/SSIM/LPIPS/DISTS/NIQE/MUSIQ/CLIPIQA）和时序指标（tOF）

EOF

# ============================================================
# 主循环
# ============================================================
for EP in "${EPOCHS[@]}"; do
    TAG="epoch_${EP}ema"
    echo "========================================"
    echo "  Evaluating $TAG ..."
    echo "========================================"

    cat >> "$OUT_MD" << EOF
---

## $TAG

EOF

    # --------------------------------------------------------
    # Vid4
    # --------------------------------------------------------
    PRED_VID4="results/${EXP}/${TAG}/Vid4_Pred"
    echo "--- Vid4 画质 ---"
    cat >> "$OUT_MD" << 'EOF'
### Vid4 — 画质指标

```
EOF
    python evaluate.py \
        --pred_dir "$PRED_VID4" \
        --gt_dir "$GT_VID4" \
        --crop_border 4 \
        | tee -a "$OUT_MD"
    echo '```' >> "$OUT_MD"

    echo "--- Vid4 tOF ---"
    cat >> "$OUT_MD" << 'EOF'

### Vid4 — 时序指标（tOF）

```
EOF
    python evaluate_tof.py \
        --pred_dir "$PRED_VID4" \
        --gt_dir "$GT_VID4" \
        --crop_border 4 \
        | tee -a "$OUT_MD"
    echo '```' >> "$OUT_MD"

    # --------------------------------------------------------
    # UDM10
    # --------------------------------------------------------
    PRED_UDM10="results/${EXP}/${TAG}/UDM10_Pred"
    echo "--- UDM10 画质 ---"
    cat >> "$OUT_MD" << 'EOF'

### UDM10 — 画质指标

```
EOF
    python evaluate.py \
        --pred_dir "$PRED_UDM10" \
        --gt_dir "$GT_UDM10" \
        --crop_border 4 \
        | tee -a "$OUT_MD"
    echo '```' >> "$OUT_MD"

    echo "--- UDM10 tOF ---"
    cat >> "$OUT_MD" << 'EOF'

### UDM10 — 时序指标（tOF）

```
EOF
    python evaluate_tof.py \
        --pred_dir "$PRED_UDM10" \
        --gt_dir "$GT_UDM10" \
        --crop_border 4 \
        | tee -a "$OUT_MD"
    echo '```' >> "$OUT_MD"

    # --------------------------------------------------------
    # REDS4
    # --------------------------------------------------------
    PRED_REDS4="results/${EXP}/${TAG}/REDS4_Pred"
    echo "--- REDS4 画质 ---"
    cat >> "$OUT_MD" << 'EOF'

### REDS4 — 画质指标

```
EOF
    python evaluate.py \
        --pred_dir "$PRED_REDS4" \
        --gt_dir "$GT_REDS4" \
        --crop_border 4 \
        | tee -a "$OUT_MD"
    echo '```' >> "$OUT_MD"

    echo "--- REDS4 tOF ---"
    cat >> "$OUT_MD" << 'EOF'

### REDS4 — 时序指标（tOF）

```
EOF
    python evaluate_tof.py \
        --pred_dir "$PRED_REDS4" \
        --gt_dir "$GT_REDS4" \
        --crop_border 4 \
        | tee -a "$OUT_MD"
    echo '```' >> "$OUT_MD"

    echo "$TAG 评估完成"
    echo ""
done

echo "========================================"
echo "全部完成！结果已写入 $OUT_MD"
echo "========================================"
