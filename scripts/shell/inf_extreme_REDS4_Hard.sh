#!/bin/bash
# 对 REDS4_Hard（极端噪声+双重JPEG压缩）数据集进行 5 模型推理对比

set -uo pipefail

REDS4_HARD="/home/ubuntu/data/REDS4_Hard"
PROJ_DIR="/home/ubuntu/lib/ST_VSR_Project"
LOG_FILE="${PROJ_DIR}/outputs/logs/inf_extreme_REDS4_Hard.log"

cd "$PROJ_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo "===== REDS4_Hard 极端退化推理 =====" | tee "$LOG_FILE"
echo "输入: $REDS4_HARD" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# 1. STAR（PNG 序列先转 .mp4，再调用 inference_sr.py）
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "[1/5] STAR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

export HF_ENDPOINT=https://hf-mirror.com

STAR_LQ_DIR="/tmp/REDS4_Hard_mp4"
mkdir -p "$STAR_LQ_DIR"
mkdir -p /home/ubuntu/data/STAR/GT_video/REDS4_Hard

for SEQ_DIR in "${REDS4_HARD}"/*/; do
    SEQ=$(basename "$SEQ_DIR")
    echo "  [STAR] 转换序列 ${SEQ} PNG → mp4" | tee -a "$LOG_FILE"
    ffmpeg -y -framerate 30 \
        -i "${SEQ_DIR}/%08d.png" \
        -c:v libx264 -pix_fmt yuv420p \
        "${STAR_LQ_DIR}/${SEQ}.mp4" \
        2>&1 | tee -a "$LOG_FILE"
done

for SEQ_DIR in "${REDS4_HARD}"/*/; do
    SEQ=$(basename "$SEQ_DIR")
    echo "  [STAR] 推理序列 ${SEQ}" | tee -a "$LOG_FILE"
    /home/ubuntu/lib/STAR/.venv/bin/python \
        /home/ubuntu/lib/STAR/video_super_resolution/scripts/inference_sr.py \
        --solver_mode fast \
        --steps 15 \
        --input_path "${STAR_LQ_DIR}/${SEQ}.mp4" \
        --model_path /home/ubuntu/data/STAR/pretrained_weight/heavy_deg.pt \
        --prompt "a good video" \
        --upscale 4 \
        --max_chunk_len 50 \
        --file_name "${SEQ}.mp4" \
        --save_dir /home/ubuntu/data/STAR/GT_video/REDS4_Hard \
        2>&1 | tee -a "$LOG_FILE"
done
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# 2. SCST（mococtrl，逐序列推理）
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "[2/5] SCST" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd /home/ubuntu/lib/SCST
for SEQ_DIR in "${REDS4_HARD}"/*/; do
    SEQ=$(basename "$SEQ_DIR")
    echo "  [SCST] 推理序列 ${SEQ}" | tee -a "$LOG_FILE"
    /home/ubuntu/lib/SCST/.venv/bin/python inference_SCST.py \
        --pretrained_model_path checkpoints/stable-diffusion-2-1-base \
        --ckpt_model_path checkpoints/mococtrl_unet.pth \
        --controlnet_path checkpoints/controlnet \
        --unet_config_path models/configs/mococtrl.yaml \
        --video_path "${SEQ_DIR}" \
        --output_dir "outputs/mococtrl_REDS4_Hard/${SEQ}" \
        --upscale 4 \
        --process_size 768 \
        --num_frame 1 \
        --overlap_frame 0 \
        --num_inference_steps 20 \
        --added_noise_level 200 \
        --init_noise_level 999 \
        --decoder_tiled_size 224 \
        --encoder_tiled_size 2048 \
        --latent_tiled_size 224 \
        --guidance_scale 5.0 \
        --seed 42 \
        --prompt "A high quality video frame, highly detailed." \
        --negative_prompt "blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed" \
        2>&1 | tee -a "$LOG_FILE"
done
cd "$PROJ_DIR"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# 3. DiffVSR（逐序列推理）
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "[3/5] DiffVSR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd /home/ubuntu/lib/DiffVSR
for SEQ_DIR in "${REDS4_HARD}"/*/; do
    SEQ=$(basename "$SEQ_DIR")
    echo "  [DiffVSR] 推理序列 ${SEQ}" | tee -a "$LOG_FILE"
    /home/ubuntu/lib/DiffVSR/.venv/bin/python inference_tile.py \
        -i "${SEQ_DIR}" \
        -o "output/REDS4_Hard" \
        -p "pretrained_models/DiffVSR_UNet.pt" \
        -oimg "output/REDS4_Hard" \
        2>&1 | tee -a "$LOG_FILE"
done
cd "$PROJ_DIR"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# 4. RealViformer（一次性处理全部序列）
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "[4/5] RealViformer" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd /home/ubuntu/lib/RealViformer
/home/ubuntu/lib/RealViformer/.venv/bin/python inference_realviformer.py \
    --model_path pretrained_model/weights.pth \
    --input_path "${REDS4_HARD}" \
    --save_path results/RealViformer_REDS4_Hard \
    --interval 100 \
    2>&1 | tee -a "$LOG_FILE"
cd "$PROJ_DIR"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------
# 5. Ours（逐序列推理，epoch_60 EMA）
# -----------------------------------------------
echo "========================================" | tee -a "$LOG_FILE"
echo "[5/5] Ours" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for SEQ_DIR in "${REDS4_HARD}"/*/; do
    SEQ=$(basename "$SEQ_DIR")
    echo "  [Ours] 推理序列 ${SEQ}" | tee -a "$LOG_FILE"
    uv run python inference.py \
        --input_dir "${SEQ_DIR}" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_60ema/REDS4_Hard_Pred/${SEQ}" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_60.pth \
        --scale 4 \
        --fps 30 \
        --use_ema \
        2>&1 | tee -a "$LOG_FILE"
done
echo "" | tee -a "$LOG_FILE"

echo "===== 全部完成: $(date) =====" | tee -a "$LOG_FILE"
