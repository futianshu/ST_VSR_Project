# ST-VSR Project

视频超分辨率训练与实验工程。核心训练、推理、评估入口保留在根目录，辅助脚本、文档和本地产物已经按用途归档。

## 常用入口

```bash
python train.py
python inference.py --help
python evaluate.py --help
python evaluate_tof.py --help
python evaluate_extreme.py --help
```

后台训练：

```bash
nohup python train.py > train.log 2>&1 &
tail -f train.log
```

## 目录约定

| 路径 | 说明 |
| --- | --- |
| `models/` | ST-VSR 网络和备用模型模块。 |
| `datasets/` | Vimeo90K 数据集读取逻辑。 |
| `utils/` | 通用工具函数。 |
| `scripts/benchmark/` | 参数量、FLOPs、runtime 统计脚本。 |
| `scripts/visualization/` | 论文对比图、任意倍率展示、气泡图脚本。 |
| `scripts/data/` | 数据生成/退化脚本。 |
| `scripts/experimental/` | 实验性入口，例如双权重融合推理。 |
| `scripts/external/` | 外部框架/基线模型调用脚本。 |
| `scripts/shell/` | 批量推理和评估 shell 脚本。 |
| `docs/` | 实验记录。 |
| `outputs/` | 本地产物：日志、可视化图、任意倍率结果等，默认被 git 忽略。 |
| `archive/` | 历史备份，默认被 git 忽略。 |

更完整的工程说明见 `PROJECT_STRUCTURE.md`。
