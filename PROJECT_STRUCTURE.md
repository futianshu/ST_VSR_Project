# ST-VSR 工程结构总结

本文档根据当前仓库结构整理，便于快速理解本视频超分辨率训练工程的模块边界、训练流程和实验资产。

## 1. 项目定位

本工程是一个视频超分辨率研究与训练项目，核心模型为 `ST_VSR_Network`。模型输入相邻 3 帧低分辨率图像，结合空间坐标和时间坐标 `coords_xyt`，输出目标时刻的高分辨率 RGB 像素点。工程围绕以下几条线展开：

- 训练：`train.py`
- 模型：`models/st_network.py`
- 数据集：`datasets/vimeo90k_st.py`
- 推理：`inference.py`、`scripts/experimental/inference2.py`
- 评估：`evaluate.py`、`evaluate_tof.py`、`evaluate_extreme.py`
- 辅助脚本：`scripts/`
- 实验记录和论文材料：`docs/records/`

## 2. 顶层目录和文件

| 路径 | 作用 |
| --- | --- |
| `train.py` | 主训练脚本，包含损失函数、GPU 退化流水线、实验路由、断点恢复、EMA、验证和保存逻辑。 |
| `models/st_network.py` | 核心 ST-VSR 网络定义，包含 SD3 VAE 语义先验、物理时空流、潜空间对齐、SFT 融合、Deformable INR 查询等。 |
| `datasets/vimeo90k_st.py` | Vimeo90K 训练/验证数据集。训练阶段输出高清三帧和坐标监督，退化在 GPU 上完成；验证阶段生成固定 Bicubic LR。 |
| `inference.py` | 标准推理入口，对 LR 序列逐帧滑窗超分，支持自动识别消融权重和 `--use_ema`。 |
| `evaluate.py` | 空域/感知/无参考指标评估，输出 PSNR(Y)、SSIM(Y)、LPIPS、DISTS、NIQE、MUSIQ、CLIPIQA。 |
| `evaluate_tof.py` | 基于 RAFT 光流的时序一致性指标 tOF 评估。 |
| `evaluate_extreme.py` | 极端运动和遮挡区域指标评估，使用 RAFT 构造快速运动掩码和遮挡掩码。 |
| `scripts/experimental/inference2.py` | 双权重融合推理实验脚本，分别加载平滑 EMA 模型和锐化 Base 模型。当前融合权重写死为只使用平滑结果。 |
| `scripts/data/generate_hard_lr.py` | 生成 REDS4_Hard 极端退化集，叠加高斯噪声和双重 JPEG 压缩。 |
| `scripts/external/run_mmagic.py` | 使用 MMagicInferencer 批量跑外部 VSR 基线模型。 |
| `scripts/benchmark/` | 参数量、FLOPs 和 runtime benchmark 脚本。 |
| `scripts/visualization/` | 论文可视化、任意倍率展示和 PSNR-runtime 气泡图脚本。 |
| `scripts/shell/` | 批量推理/评估 shell 脚本。 |
| `docs/` | 实验记录。 |
| `outputs/` | 本地生成的日志、图片、任意倍率结果等产物，默认不纳入 git。 |
| `pyproject.toml` / `uv.lock` | Python/uv 依赖配置，主要依赖包括 PyTorch、diffusers、PEFT、kornia、lpips、pyiqa、safetensors 等。 |
| `README.md` | 简短训练启动说明。 |
| `CLAUDE.md` | 先前整理的工程说明和常用命令。 |

## 3. 代码目录

### `models/`

| 文件 | 说明 |
| --- | --- |
| `st_network.py` | 当前主模型。 |

`ST_VSR_Network` 的核心组件：

- `AutoencoderKL`：从本地 `stabilityai/stable-diffusion-3-medium-diffusers/vae` 加载 SD3 VAE，作为冻结语义先验提取器。
- LoRA adapter：通过 PEFT 注入到 VAE 指定模块，训练/加载 DPAS-SR 风格 VAE 先验权重。
- `pre_cleaner`：VAE 编码前的浅层去噪/去压缩伪影模块，初始化为近似恒等映射。
- `TSM_ResBlock`：物理流使用的零参数时间偏移残差块，用于 3 帧时序交互。
- `LatentAlign_Block`：在 VAE latent 空间中对前后帧向当前帧做对齐。
- `SFT_Layer`：使用语义 latent 特征调制物理流特征。
- `offset_conv`：预测空间坐标偏移，支撑 Deformable INR 查询。
- `PositionalEncoding3D` + `inr_mlp`：对 `(x, y, t)` 做位置编码，并解码 RGB 残差。
- `_forward_chunk`：支持按 `chunk_size` 分块查询，推理时降低显存占用。

### `datasets/`

| 文件 | 说明 |
| --- | --- |
| `vimeo90k_st.py` | Vimeo90K septuplet 数据集定义。 |

训练集 `Vimeo90K_ST_Dataset`：

- 读取 `sep_trainlist.txt`。
- 随机抽取间隔为 2 的 3 帧，例如 `im1/im3/im5`。
- 随机选择目标时间 `t ∈ {-0.5, 0.0, 0.5}`，分别对应输入帧之间的插帧/中心帧超分目标。
- 做水平翻转、垂直翻转、时序反转。
- 对高清 patch 生成坐标监督 `coords_xyt` 和目标 RGB 点 `gt_rgb_points`。
- 不在 CPU 上退化，训练时把干净 HR patch 交给 `train.py` 的 GPU 退化流水线。

验证集 `Vimeo90K_ST_Val_Dataset`：

- 读取 `sep_testlist.txt`。
- 默认最多均匀采样 500 个片段。
- 固定输入 `im2/im4/im6`，目标为 `im4`，时间 `t=0.0`。
- 使用 Bicubic 生成 LR，做全分辨率验证。

### `utils/`

| 文件 | 说明 |
| --- | --- |
| `util.py` | 参数查看、LoRA 权重智能匹配加载、可训练参数名收集。 |

## 4. 训练流程

主入口为 `train.py`。

### 实验路由

通过文件顶部的 `EXP_NAME` 切换实验：

- `ablation_wo_time_cond`：关闭时间条件。
- `ablation_wo_shallow`：关闭物理浅层 CNN。
- `ablation_wo_semantic_prior`：关闭 VAE 语义先验。
- `full_model` / `full_model_tmax30`：完整模型变体。
- `Ours_DeformINR_LatentPrior_Ep55`：当前主要结果目录名。

### 数据和先验路径

训练脚本中写死的主要路径：

- Vimeo90K：`/home/ubuntu/data/OpenDataLab___Vimeo90K/raw/vimeo_septuplet`
- VAE LoRA 先验：`/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors`

### 损失函数

`train.py` 内定义并使用：

- `CharbonnierLoss`：基础重建损失。
- `LPIPS(VGG)`：感知损失，40 轮后启用。
- `FocalFrequencyLoss`：频域焦点损失，40 轮后启用。
- `PatchGANDiscriminator` + BCE：2D 条件时空判别器，输入为前帧、目标帧、后帧在通道维拼接后的 9 通道图像。

### 退化和训练阶段

训练数据集输出的是干净 HR 三帧。`train.py` 内部的 `GPUDegradation` 在 GPU 上做实时退化：

- Epoch 1-40：强退化，偏向打牢物理重建基础。
- Epoch 41-70：弱退化，并启用感知、对抗和频域损失。
- 退化包含高斯模糊、Bicubic 下采样、高斯噪声、JPEG 压缩。

训练配置要点：

- Batch size 当前为 16。
- 优化器为 AdamW。
- 前 40 轮使用 Linear Warmup + Cosine Annealing。
- 40 轮后进入对抗微调，生成器/判别器使用更低学习率。
- 使用 AMP 混合精度、梯度裁剪、EMA。
- 使用 `torch.multiprocessing.set_sharing_strategy('file_system')` 和 OpenCV 线程关闭来规避共享内存/多线程问题。
- 大量 FP32 强制块和 contiguous hook 用于 V100 兼容和数值稳定。

### 断点和保存

当前脚本默认 `resume_epoch = 40`，会尝试从：

```bash
checkpoints/${EXP_NAME}/st_vsr_epoch_40.pth
```

恢复。每轮保存：

- `st_vsr_latest.pth`
- `st_vsr_best.pth`
- `st_vsr_epoch_{epoch}.pth`

保存时会剥离 `encoder` 权重，避免把冻结 SD3 VAE 大权重写入检查点。

## 5. 推理流程

### 标准推理

入口：`inference.py`

常用命令：

```bash
python inference.py \
  --input_dir <LR序列目录> \
  --output_dir results/<实验名>/<数据集>_Pred/<序列名> \
  --checkpoint checkpoints/Ours_DeformINR_LatentPrior_Ep55/st_vsr_epoch_60.pth \
  --scale 4 \
  --fps 30 \
  --use_ema
```

关键逻辑：

- 自动根据 checkpoint 路径识别消融实验类型，并对应关闭时间条件、物理流或语义先验。
- 读取 LR 序列，按当前帧 `i` 构造 `[i-2, i, i+2]` 的 3 帧窗口，边界用首尾帧补齐。
- 根据 `--scale` 生成 HR 坐标，所以支持非整数或任意倍率可视化实验。
- `--chunk_size` 控制 INR 坐标分块，默认 30000。
- 同时保存逐帧 PNG/JPG 和 `output_video.mp4`。
- `--use_ema` 加载 `ema_model_state_dict`，否则加载 `model_state_dict`。

### 双权重融合推理

入口：`scripts/experimental/inference2.py`

用途是同时加载平滑底盘模型和锐化特征模型做融合实验。当前代码中：

```python
pred_fused = pred_smooth * 1 + pred_sharp * 0
```

也就是实际只输出平滑模型结果。

## 6. 评估体系

### 画质和感知指标

入口：`evaluate.py`

输出指标：

- PSNR(Y)
- SSIM(Y)
- LPIPS
- DISTS
- NIQE
- MUSIQ
- CLIPIQA

脚本支持输入按序列组织的目录，也能兼容单序列目录。默认 `--crop_border 4`。

### 时序一致性

入口：`evaluate_tof.py`

使用 `torchvision.models.optical_flow.raft_small` 根据 GT 估计光流，将上一帧预测 warp 到当前帧，再计算 MSE x100，作为 tOF。数值越低代表闪烁越少。

### 极端运动/遮挡

入口：`evaluate_extreme.py`

使用 RAFT 前向/后向光流：

- 光流幅值大于 `motion_thresh` 的区域为快速运动区域。
- 前后向光流一致性差的区域为遮挡区域。
- 分别统计这些区域上的 PSNR(Y) 和 SSIM(Y)。

## 7. 批处理脚本

`scripts/shell/` 下主要是实验批处理脚本：

- `inf_45_50_55_60_65_70.sh`：对多个 epoch 的 EMA 模型在 Vid4、UDM10、REDS4 上批量推理。
- `evaluate_45_50_55_60_65_70.sh`：对上述推理结果批量评估，并写入 `results/metrics/eval_45_50_55_60_65_70.md`。
- `inf_ablation_wo_time_cond_ep60.sh` / `eval_ablation_wo_time_cond_ep60.sh`：无时间条件消融的推理和评估。
- `inf_ablation_wo_semantic_prior_ep60.sh` / `eval_ablation_wo_semantic_prior_ep60.sh`：无语义先验消融的推理和评估。
- `inf_extreme_REDS4_Hard.sh` / `eval_extreme_REDS4_Hard.sh`：REDS4_Hard 上本文模型和外部基线的推理、评估。
- `evaluate_extreme_REDS4.sh`：对 REDS4 上多个方法做极端运动/遮挡评估。

这些脚本中大量使用本机绝对路径，迁移环境时需要优先检查数据集、外部模型和虚拟环境路径。

## 8. 实验产物和记录

| 目录 | 说明 |
| --- | --- |
| `checkpoints/` | 模型权重，按实验名分组，例如 `Ours_DeformINR_LatentPrior_Ep55/`、`ablation_wo_time_cond/` 等。 |
| `results/` | 推理结果、基线结果、评估 Markdown 报告。 |
| `outputs/logs/` | 训练或运行日志。 |
| `docs/records/` | 手工整理的实验记录、参考文献性能、本文结果表。 |
| `outputs/visual_comparisons/` | 可视化对比图。 |
| `outputs/arbitrary_scale_REDS4/` | REDS4 任意倍率推理和可视化产物。 |
| `outputs/arbitrary_scale_UDM10/` | UDM10 任意倍率推理和可视化产物。 |
| `archive/bak/` | 历史备份目录。 |

注意：`.gitignore` 已忽略 `checkpoints/`、`results/`、`outputs/`、`archive/`、`.venv/`、`.venv-mmagic/`、`*.log` 等，避免本地大文件和一次性实验产物继续堆到 git 状态里。

## 9. 数据流概览

### 训练

```text
Vimeo90K HR septuplet
  -> Vimeo90K_ST_Dataset 抽取 3 帧 HR + 目标时间 t + HR 坐标监督
  -> GPUDegradation 生成 LR 3 帧
  -> ST_VSR_Network(lr_seq, coords_xyt)
  -> 输出目标 HR RGB points
  -> Charbonnier / LPIPS / GAN / FFL
  -> EMA 验证
  -> checkpoints/<EXP_NAME>/
```

### 推理

```text
LR 图像序列目录
  -> inference.py 按 [i-2, i, i+2] 构造 3 帧窗口
  -> 根据 scale 生成 HR 坐标
  -> ST_VSR_Network 分块查询 INR
  -> 保存 HR 帧和 output_video.mp4
```

### 评估

```text
results/<实验>/<数据集>_Pred/
  -> evaluate.py      画质/感知/无参考指标
  -> evaluate_tof.py  时序一致性 tOF
  -> evaluate_extreme.py 极端运动/遮挡区域指标
  -> results/metrics/*.md 或 *.log
```

## 10. 当前工程注意点

- `train.py` 的 `EXP_NAME`、`resume_epoch`、数据路径和 VAE 先验路径都是硬编码的，换实验前要同步检查。
- `ST_VSR_Network` 默认从本地 Hugging Face 缓存加载 SD3 VAE，`local_files_only=True`，新机器上需要提前准备模型缓存。
- 推理脚本会根据 checkpoint 路径字符串自动切换消融配置，权重目录命名会影响模型实例化。
- 训练脚本保存 checkpoint 时不保存 encoder 权重，加载时需要再次调用 `load_dpas_sr_prior` 注入 VAE LoRA 先验。
- 多个脚本依赖外部仓库路径，例如 `/home/ubuntu/lib/SCST`、`/home/ubuntu/lib/DiffVSR`、`/home/ubuntu/lib/RealViformer`、`/home/ubuntu/lib/STAR`。
- 评估脚本中 `pyiqa` 的部分指标被放到 CPU 上运行，以规避 V100 上的矩阵乘法兼容问题。
- 当前根目录已按“核心入口、代码模块、辅助脚本、文档记录、本地产物”重新归类；后续继续重构前建议先确认已有未提交代码改动是否都属于当前实验。
