#!/bin/bash
# 对 5 个模型在 REDS4_Hard（极端噪声+双重JPEG压缩）上的推理结果进行 7 维指标评估

set -uo pipefail

GT_DIR="/home/ubuntu/data/REDS4/train_sharp"
PROJ_DIR="/home/ubuntu/lib/ST_VSR_Project"
LOG_FILE="${PROJ_DIR}/outputs/logs/eval_extreme_REDS4_Hard.log"

cd "$PROJ_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo "===== REDS4_Hard 极端退化评估 =====" | tee "$LOG_FILE"
echo "GT: $GT_DIR" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# STAR：mp4 抽帧 → PNG，再评估
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "模型: STAR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

STAR_PNG_DIR="/home/ubuntu/data/STAR/GT_video/REDS4_Hard_png"
for mp4 in /home/ubuntu/data/STAR/GT_video/REDS4_Hard/*.mp4; do
    SEQ=$(basename "$mp4" .mp4)
    OUT_SEQ="${STAR_PNG_DIR}/${SEQ}"
    mkdir -p "$OUT_SEQ"
    echo "  [STAR] 抽帧 ${SEQ}.mp4 → PNG" | tee -a "$LOG_FILE"
    ffmpeg -y -i "$mp4" -start_number 0 "${OUT_SEQ}/%08d.png" \
        2>&1 | tee -a "$LOG_FILE"
done

uv run python evaluate.py \
    --pred_dir "$STAR_PNG_DIR" \
    --gt_dir   "$GT_DIR" \
    --crop_border 4 \
    2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# SCST
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "模型: SCST" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

uv run python evaluate.py \
    --pred_dir "/home/ubuntu/lib/SCST/outputs/mococtrl_REDS4_Hard" \
    --gt_dir   "$GT_DIR" \
    --crop_border 4 \
    2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# DiffVSR
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "模型: DiffVSR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

uv run python evaluate.py \
    --pred_dir "/home/ubuntu/lib/DiffVSR/output/REDS4_Hard" \
    --gt_dir   "$GT_DIR" \
    --crop_border 4 \
    2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# RealViformer
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "模型: RealViformer" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

uv run python evaluate.py \
    --pred_dir "/home/ubuntu/lib/RealViformer/results/RealViformer_REDS4_Hard" \
    --gt_dir   "$GT_DIR" \
    --crop_border 4 \
    2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# Ours
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "模型: Ours" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

uv run python evaluate.py \
    --pred_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_60ema/REDS4_Hard_Pred" \
    --gt_dir   "$GT_DIR" \
    --crop_border 4 \
    2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "===== 全部完成: $(date) =====" | tee -a "$LOG_FILE"
