#!/bin/bash
cd /home/ubuntu/lib/ST_VSR_Project

################################################
# Vid4
################################################

# epoch_45ema
for seq in /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_45ema/Vid4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_45.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_50ema
for seq in /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_50ema/Vid4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_50.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_55ema
for seq in /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_55ema/Vid4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_55.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_60ema
for seq in /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_60ema/Vid4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_60.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_65ema
for seq in /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_65ema/Vid4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_65.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_70ema
for seq in /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_70ema/Vid4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_70.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

################################################
# UDM10
################################################

# epoch_45ema
for seq in /home/ubuntu/data/UDM10/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_45ema/UDM10_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_45.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_50ema
for seq in /home/ubuntu/data/UDM10/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_50ema/UDM10_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_50.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_55ema
for seq in /home/ubuntu/data/UDM10/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_55ema/UDM10_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_55.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_60ema
for seq in /home/ubuntu/data/UDM10/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_60ema/UDM10_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_60.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_65ema
for seq in /home/ubuntu/data/UDM10/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_65ema/UDM10_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_65.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_70ema
for seq in /home/ubuntu/data/UDM10/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_70ema/UDM10_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_70.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

################################################
# REDS4
################################################

# epoch_45ema
for seq in /home/ubuntu/data/REDS4/train_sharp_bicubic/X4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_45ema/REDS4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_45.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_50ema
for seq in /home/ubuntu/data/REDS4/train_sharp_bicubic/X4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_50ema/REDS4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_50.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_55ema
for seq in /home/ubuntu/data/REDS4/train_sharp_bicubic/X4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_55ema/REDS4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_55.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_60ema
for seq in /home/ubuntu/data/REDS4/train_sharp_bicubic/X4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_60ema/REDS4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_60.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_65ema
for seq in /home/ubuntu/data/REDS4/train_sharp_bicubic/X4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_65ema/REDS4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_65.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done

# epoch_70ema
for seq in /home/ubuntu/data/REDS4/train_sharp_bicubic/X4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_70ema/REDS4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_70.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done
