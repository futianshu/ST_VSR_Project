# Scripts

辅助脚本按用途分组，核心入口仍在项目根目录。

| 子目录 | 用途 |
| --- | --- |
| `benchmark/` | 参数量、FLOPs、runtime 统计。 |
| `visualization/` | 论文可视化、任意倍率展示、气泡图。 |
| `data/` | 数据生成和退化。 |
| `experimental/` | 临时或探索性推理入口。 |
| `external/` | 外部框架和基线模型调用。 |
| `shell/` | 批量推理、评估和消融实验脚本。 |

从仓库根目录运行脚本最稳妥，例如：

```bash
python scripts/benchmark/profile_model.py
bash scripts/shell/evaluate_45_50_55_60_65_70.sh
```
