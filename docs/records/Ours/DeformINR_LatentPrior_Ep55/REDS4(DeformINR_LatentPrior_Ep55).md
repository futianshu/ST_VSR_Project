#### 推理

```shell
# epoch_40ema
for seq in /home/ubuntu/data/REDS4/train_sharp_bicubic/X4/*; do
    seq_name=$(basename $seq)
    python inference.py \
        --input_dir "$seq" \
        --output_dir "results/Ours_DeformINR_LatentPrior_Ep55/epoch_40ema/REDS4_Pred/$seq_name" \
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
    --pred_dir results/Ours_DeformINR_LatentPrior_Ep55/epoch_40ema/REDS4_Pred \
    --gt_dir /home/ubuntu/data/REDS4/train_sharp \
    --crop_border 4
    
--------------------------------------------------------------------------------------------------
Sequence     | Frames | PSNR(Y)↑ | SSIM(Y)↑ | LPIPS↓   | DISTS↓   | NIQE↓    | MUSIQ↑   | CLIPIQA↑
--------------------------------------------------------------------------------------------------
000          | 100    |    27.05 |   0.7509 |   0.3308 |   0.1578 |   5.8384 |  51.7572 |   0.3405
011          | 100    |    28.84 |   0.8064 |   0.3498 |   0.1514 |   6.1927 |  42.6393 |   0.2424
015          | 100    |    31.72 |   0.8768 |   0.3033 |   0.1617 |   6.1839 |  49.6441 |   0.2510
020          | 100    |    28.01 |   0.8165 |   0.3279 |   0.1483 |   6.3569 |  46.6501 |   0.2484
--------------------------------------------------------------------------------------------------
OVERALL      | 400    |    28.91 |   0.8127 |   0.3280 |   0.1548 |   6.1430 |  47.6727 |   0.2706
--------------------------------------------------------------------------------------------------
```

#### 评估光流

```shell
# epoch_40ema
python evaluate_tof.py \
    --pred_dir results/Ours_DeformINR_LatentPrior_Ep55/epoch_40ema/REDS4_Pred \
    --gt_dir /home/ubuntu/data/REDS4/train_sharp \
    --crop_border 4

--------------------------------------------------
Sequence        | Pairs  | tOF (MSE x100) ↓
--------------------------------------------------
000             | 99     |     0.0686
011             | 99     |     0.0859
015             | 99     |     0.2026
020             | 99     |     0.1931
--------------------------------------------------
OVERALL         | 396    |     0.1375
--------------------------------------------------
```
