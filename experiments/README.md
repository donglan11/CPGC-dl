# experiments —— 消融与对比实验脚本

论文中全部对照实验的原始脚本，共 **60 个**，按功能分六类。这些脚本保留原始形态（未做路径参数化），阅读时可对照 `results/README.md` 的口径说明。

> ⚠️ 脚本内的 `DATA_DIR`、`DINO_WEIGHTS_PATH` 等路径是作者本机的绝对路径，运行前需自行修改，或参考根目录 `train.py` 的参数化写法。

## 目录

| 目录 | 个数 | 内容 |
|---|---|---|
| `core/` | 5 | 核心训练脚本：10 种子 / 3 种子训练、单种子日志、调参模板 |
| `ablation/` | 21 | 消融变体：Label Smoothing、Mixup、CutMix、Dropout、多原型、LDAM、类加权 SupCon、λ_proto 调优、批次大小敏感性、去采样器、去 SupCon、去 DINO、TTA，以及生成这些变体的脚本 |
| `analysis/` | 9 | 机制与敏感性分析：极端 α、inner-val 敏感性、Grad-CAM 融合机制核查、t-SNE 紧凑度/分离度、批量与 num_workers 基准、一致性核对 |
| `baselines/` | 9 | 基线与对比方法：ResNet-50 基线、SE、CBAM、SimAM、冻结骨干实验、CPGC vs SE 显著性检验 |
| `visualization/` | 7 | Grad-CAM、t-SNE、混淆矩阵、训练曲线、参数量与速度对比图 |
| `stats/` | 9 | 统计检验与逐类分析：F1/Precision/Recall 计算、逐类配对 t 检验、逐种子导出、逐类精度对比图 |

## 关键实验与结论对应

| 论文结论 | 脚本 |
|---|---|
| 主表 91.76 vs 89.81 | `core/CPGC_10种子训练.py` |
| LOO 消融：w/o DINO −2.01、w/o 原型融合 −0.64 | `ablation/CPGC_noDINO_3种子训练.py`、`ablation/CPGC_noSC_3种子训练.py`（及对应原型消融） |
| 可学习融合必要性（极端 α） | `analysis/CPGC_极端alpha分析.py` |
| 训练稳定性：去 SupCon 后方差 0.25→1.46 | `ablation/CPGC_noSC_3种子训练.py` + `stats/` |
| 冻结骨干 +5.58 | `baselines/GC10_冻结版*.py` |
| 跨数据集泛化（X-SDD） | `baselines/` 与 `X-SDD实验/`（未纳入本仓库，数据集不同） |
| 逐类显著性（Bonferroni / FDR） | `stats/逐类分析_10种子.py` |
| 参数量与推理开销 | `visualization/画参数量与速度对比图.py` |

## 负结果同样保留

以下实验**未超过完整 CPGC**，论文如实报告：

- Label Smoothing、Mixup、CutMix、Dropout、多原型：均无增益
- TTA：−1.43 pp（反而下降）
- 原始 EMA / HDC-EMA 注意力路线：与原始 EMA 无显著差异
- 组合策略（Focal + CutMix + Sampler）：−8.51 pp（多重校正叠加导致训练分布失真）

保留负结果是这个仓库的方法论之一——只报告成功会让别人无法判断哪些路走不通。
