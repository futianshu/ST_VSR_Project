#!/bin/bash
# 对 ablation_wo_time_cond epoch_60ema 在 Vid4、UDM10、REDS4 上进行评估
# 评估结果写入 results/metrics/eval_ablation_wo_time_cond_ep60.md

set -e
cd /home/ubuntu/lib/ST_VSR_Project

EXP="ablation_wo_time_cond/epoch_60ema"
OUT_MD="results/metrics/eval_ablation_wo_time_cond_ep60.md"

GT_VID4="/home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/GT"
GT_UDM10="/home/ubuntu/data/UDM10/GT"
GT_REDS4="/home/ubuntu/data/REDS4/train_sharp"

mkdir -p results/metrics

cat > "$OUT_MD" << 'EOF'
# ablation_wo_time_cond — epoch_60ema 评估结果

> 评估数据集：Vid4 / UDM10 / REDS4，超分倍率 4×，`--crop_border 4`
> 包含画质指标（PSNR/SSIM/LPIPS/DISTS/NIQE/MUSIQ/CLIPIQA）和时序指标（tOF）

---

EOF

# --------------------------------------------------------
# Vid4
# --------------------------------------------------------
PRED_VID4="results/${EXP}/Vid4_Pred"

cat >> "$OUT_MD" << 'EOF'
## Vid4 — 画质指标

```
EOF
python evaluate.py \
    --pred_dir "$PRED_VID4" \
    --gt_dir "$GT_VID4" \
    --crop_border 4 \
    | tee -a "$OUT_MD"
echo '```' >> "$OUT_MD"

cat >> "$OUT_MD" << 'EOF'

## Vid4 — 时序指标（tOF）

```
EOF
python evaluate_tof.py \
    --pred_dir "$PRED_VID4" \
    --gt_dir "$GT_VID4" \
    --crop_border 4 \
    | tee -a "$OUT_MD"
echo '```' >> "$OUT_MD"

echo "---" >> "$OUT_MD"
echo "" >> "$OUT_MD"

# --------------------------------------------------------
# UDM10
# --------------------------------------------------------
PRED_UDM10="results/${EXP}/UDM10_Pred"

cat >> "$OUT_MD" << 'EOF'
## UDM10 — 画质指标

```
EOF
python evaluate.py \
    --pred_dir "$PRED_UDM10" \
    --gt_dir "$GT_UDM10" \
    --crop_border 4 \
    | tee -a "$OUT_MD"
echo '```' >> "$OUT_MD"

cat >> "$OUT_MD" << 'EOF'

## UDM10 — 时序指标（tOF）

```
EOF
python evaluate_tof.py \
    --pred_dir "$PRED_UDM10" \
    --gt_dir "$GT_UDM10" \
    --crop_border 4 \
    | tee -a "$OUT_MD"
echo '```' >> "$OUT_MD"

echo "---" >> "$OUT_MD"
echo "" >> "$OUT_MD"

# --------------------------------------------------------
# REDS4
# --------------------------------------------------------
PRED_REDS4="results/${EXP}/REDS4_Pred"

cat >> "$OUT_MD" << 'EOF'
## REDS4 — 画质指标

```
EOF
python evaluate.py \
    --pred_dir "$PRED_REDS4" \
    --gt_dir "$GT_REDS4" \
    --crop_border 4 \
    | tee -a "$OUT_MD"
echo '```' >> "$OUT_MD"

cat >> "$OUT_MD" << 'EOF'

## REDS4 — 时序指标（tOF）

```
EOF
python evaluate_tof.py \
    --pred_dir "$PRED_REDS4" \
    --gt_dir "$GT_REDS4" \
    --crop_border 4 \
    | tee -a "$OUT_MD"
echo '```' >> "$OUT_MD"

echo "========================================"
echo "评估完成！结果已写入 $OUT_MD"
echo "========================================"
