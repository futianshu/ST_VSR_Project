# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此代码仓库中工作时提供指导。

## 项目概述

ST-VSR 是一个时空视频超分辨率研究项目，通过双流架构实现 4 倍超分：
- **生成流**：Stable Diffusion 3 VAE 编码器（结合 PEFT/LoRA）用于语义特征提取
- **物理流**：CNN + 时间偏移模块（TSM）用于相邻帧间的特征对齐
- **输出头**：隐式神经表示（INR）MLP 在任意空间坐标处解码隐特征

模型输入为 3 帧相邻低分辨率帧（前帧、当前帧、后帧），输出中间帧的 4 倍超分结果。

## 环境配置

```bash
# 使用 uv 包管理器，需要 Python 3.13
uv sync
```

主要依赖：`torch>=2.10.0`、`diffusers>=0.37.0`、`peft>=0.18.1`、`kornia>=0.8.2`、`lpips`、`pyiqa`、`safetensors`。

## 常用命令

**训练：**
```bash
python train.py
# 或通过 nohup 在 SSH 断开后持续运行：
nohup python train.py > train.log 2>&1 &
tail -f train.log
```

通过 [train.py](train.py) 顶部的 `EXP_NAME` 变量配置实验名称。数据集路径：`/home/ubuntu/data/OpenDataLab___Vimeo90K/raw/vimeo_septuplet`。SD3 VAE 权重路径：`/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors`。

**推理：**
```bash
python inference.py \
  --input_dir <低分辨率帧目录> \
  --output_dir results/ \
  --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_best.pth \
  --scale 4.0 \
  --use_ema
```

**评估：**
```bash
python evaluate.py \
  --pred_dir <生成的高分辨率帧目录> \
  --gt_dir <真实高分辨率帧目录> \
  --crop_border 4
```

## 架构说明

### 核心模型（[models/st_network.py](models/st_network.py)）

`ST_VSR_Network` 包含以下模块：
1. **预清洗器（Pre-Cleaner）**：浅层卷积，在 VAE 编码前去除压缩/噪声伪影
2. **SD3 VAE 编码器**（冻结骨干网络 + PEFT LoRA 适配器）：提取隐特征，输出为 4 通道、8 倍下采样的隐变量
3. **隐空间对齐（Latent Alignment）**：在融合前对跨帧特征进行隐空间对齐
4. **SFT 调制**：语义特征变换（Semantic Feature Transform），以 VAE 隐变量为条件对物理流特征进行缩放和偏移
5. **INR MLP**：在查询坐标 `(x, y, t)` 处解码逐像素特征，支持任意分辨率和时间插值
6. **可变形坐标**：通过 `offset_conv` 学习 `(Δx, Δy)` 偏移量，在 INR 查询前应用

物理流使用带 TSM（时间偏移模块）的浅层 CNN，实现零参数跨帧时序特征融合。

### 数据集（[datasets/vimeo90k_st.py](datasets/vimeo90k_st.py)）

从 Vimeo90K septuplet 加载 3 帧三元组，在 GPU 上实时进行退化操作（模糊、噪声、JPEG 压缩，使用 Kornia），并随机进行水平/垂直翻转和时序翻转。时间坐标 `t ∈ {-0.5, 0.0, 0.5}` 标记帧的时间位置。

### 训练（[train.py](train.py)）

联合损失：Charbonnier（像素级）+ LPIPS（感知）+ 焦点频率损失 + PatchGAN 对抗损失。使用 AMP（FP16 自动混合精度）和 EMA 权重平均。按验证集 PSNR 保存最优检查点。

### 消融实验检查点

| 名称 | 说明 |
|------|------|
| `ablation_wo_time_cond/` | 去除时间坐标条件 |
| `ablation_wo_shallow/` | 去除物理流（CNN） |
| `full_model/` | 双流完整模型，标准训练 |
| `Ours_DeformINR_LatentPrior_Ep55/` | 最优模型：可变形 INR + 隐先验，训练 70 轮 |

## 注意事项

- 代码中包含大量中文注释
- GPU 多进程使用 `file_system` 共享策略（在 train.py 中设置），以避免 `/dev/shm` 溢出
- V100 兼容性：通过 `register_forward_pre_hook` 强制张量连续性，解决非连续张量操作问题
- 推理时 INR 坐标查询按块处理（`chunk_size=30000`）以管理 GPU 显存
