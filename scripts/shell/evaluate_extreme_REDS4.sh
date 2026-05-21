#!/bin/bash
# 对 5 个模型的 REDS4 推理结果进行极端运动/遮挡场景评估

GT_DIR="/home/ubuntu/data/REDS4/train_sharp"
LOG_FILE="outputs/logs/evaluate_extreme_REDS4.log"

declare -A METHODS
METHODS["STAR"]="/home/ubuntu/data/STAR/GT_video/REDS4_png"
METHODS["SCST"]="/home/ubuntu/lib/SCST/outputs/mococtrl_REDS4"
METHODS["DiffVSR"]="/home/ubuntu/lib/DiffVSR/output/REDS4"
METHODS["RealViformer"]="/home/ubuntu/lib/RealViformer/results/RealViformer_REDS4"
METHODS["Ours"]="/home/ubuntu/lib/ST_VSR_Project/results/Ours_DeformINR_LatentPrior_Ep55/epoch_60ema/REDS4_Pred"

ORDER=("STAR" "SCST" "DiffVSR" "RealViformer" "Ours")

cd /home/ubuntu/lib/ST_VSR_Project
mkdir -p "$(dirname "$LOG_FILE")"

echo "===== REDS4 极端运动/遮挡评估 =====" | tee "$LOG_FILE"
echo "GT: $GT_DIR" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for NAME in "${ORDER[@]}"; do
    PRED_DIR="${METHODS[$NAME]}"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "模型: $NAME" | tee -a "$LOG_FILE"
    echo "预测目录: $PRED_DIR" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    uv run python evaluate_extreme.py \
        --pred_dir "$PRED_DIR" \
        --gt_dir "$GT_DIR" \
        --motion_thresh 10.0 \
        2>&1 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
done

echo "===== 全部完成: $(date) =====" | tee -a "$LOG_FILE"
