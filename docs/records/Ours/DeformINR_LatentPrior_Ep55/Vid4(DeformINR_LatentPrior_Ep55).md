#### 推理

```shell
# epoch_40ema
for seq in /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/BIx4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_40ema/Vid4_Pred/$seq_name" \
        --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_40.pth \
        --scale 4 \
        --fps 30 \
        --use_ema
done
```

#### 评估画质

```shell
# epoch_40ema
python evaluate.py \
    --pred_dir results/Ours_DeformINR_LatentPrior_Ep55/epoch_40ema/Vid4_Pred \
    --gt_dir /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/GT \
    --crop_border 4
    
--------------------------------------------------------------------------------------------------
Sequence     | Frames | PSNR(Y)↑ | SSIM(Y)↑ | LPIPS↓   | DISTS↓   | NIQE↓    | MUSIQ↑   | CLIPIQA↑
--------------------------------------------------------------------------------------------------
calendar     | 41     |    22.02 |   0.7224 |   0.3942 |   0.2005 |   5.6599 |  58.1304 |   0.2943
city         | 34     |    26.17 |   0.7171 |   0.4035 |   0.2298 |   7.0353 |  55.0252 |   0.1985
foliage      | 49     |    24.55 |   0.6901 |   0.4055 |   0.2073 |   7.8983 |  44.7264 |   0.2995
walk         | 47     |    28.34 |   0.8790 |   0.2517 |   0.1211 |   6.7353 |  46.5747 |   0.2370
--------------------------------------------------------------------------------------------------
OVERALL      | 171    |    25.31 |   0.7552 |   0.3601 |   0.1864 |   6.8704 |  50.4960 |   0.2610
--------------------------------------------------------------------------------------------------
```

#### 评估光流

```shell
# epoch_40ema
python evaluate_tof.py \
    --pred_dir results/Ours_DeformINR_LatentPrior_Ep55/epoch_40ema/Vid4_Pred \
    --gt_dir /home/ubuntu/data/OpenDataLab___Vid4/raw/Vid4/Vid4/GT \
    --crop_border 4

--------------------------------------------------
Sequence        | Pairs  | tOF (MSE x100) ↓
--------------------------------------------------
calendar        | 40     |     0.0726
city            | 33     |     0.0447
foliage         | 48     |     0.1135
walk            | 46     |     0.3406
--------------------------------------------------
OVERALL         | 167    |     0.1527
--------------------------------------------------
```
