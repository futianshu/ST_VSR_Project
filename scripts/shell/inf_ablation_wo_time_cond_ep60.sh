################################################
# ablation_wo_time_cond — epoch_60ema
# checkpoint: checkpoints/ablation_wo_time_cond/st_vsr_epoch_60.pth
################################################

set -e
cd /home/ubuntu/lib/ST_VSR_Project

CKPT="checkpoints/ablation_wo_time_cond/st_vsr_epoch_60.pth"
EXP="ablation_wo_time_cond/epoch_60ema"

################################################
# Vid4
################################################
for seq in /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/${EXP}/Vid4_Pred/$seq_name" \
        --checkpoint "$CKPT" \
        --scale 4 \
        --fps 30 \
        --use_ema
done

################################################
# UDM10
################################################
for seq in /home/ubuntu/data/UDM10/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/${EXP}/UDM10_Pred/$seq_name" \
        --checkpoint "$CKPT" \
        --scale 4 \
        --fps 30 \
        --use_ema
done

################################################
# REDS4
################################################
for seq in /home/ubuntu/data/REDS4/train_sharp_bicubic/X4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/${EXP}/REDS4_Pred/$seq_name" \
        --checkpoint "$CKPT" \
        --scale 4 \
        --fps 30 \
        --use_ema
done

echo "========================================"
echo "推理完成！结果已保存至 results/${EXP}"
echo "========================================"
