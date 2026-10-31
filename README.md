# CPGC

**Prototype-Guided Contrastive Learning for Long-Tailed Classification of Steel Surface Defects**
原型引导对比学习的钢材表面缺陷长尾分类

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 这是什么

工业视觉检测中，钢材表面缺陷数据呈**典型长尾分布**——常见缺陷样本上千张，罕见缺陷只有几张，而罕见缺陷恰恰是最需要检出的。直接用监督学习训练，模型会把容量都花在头部类别上。

**CPGC** 把两条路径端到端融合：

```
                    ┌─────────────────────────────────────┐
   图像 x ──► 骨干 f_θ ──► h (2048 维)
              (DINO 预训练            ┌──────────────┴──────────────┐
               ResNet-50)             ▼                             ▼
                        ┌────────────────────┐        ┌────────────────────┐
                        │ 判别式分类路径      │        │ 原型检索路径        │
                        │ MLP 分类头          │        │ 投影头 2048→512→128│
                        │ s_mlp               │        │ s_proto（检索类别原型）│
                        └─────────┬──────────┘        └──────────┬─────────┘
                                  │     可学习融合权重 α = σ(θ_α)    │
                                  └───────────────┬────────────────┘
                                                  ▼
                                    s_final = α·s_mlp + (1−α)·s_proto
```

三个关键设计：

1. **有监督对比学习（SupCon）** 把投影嵌入空间塑造成"类内聚拢、类间分离"，为原型提供稳定的判别性锚点
2. **类别原型 + EMA 在线更新**（动量 m=0.9，detach 无梯度回流），在样本稀缺的少数类上不受单批次噪声干扰
3. **可学习融合权重 α**，让模型自己在训练中决定"信分类头还是信原型"——消融证明强制走任一单分支都会掉点（p<0.05），说明这个设计是必要的而非手工调参

## 结果

GC10-DET 数据集（10 类钢材表面缺陷，训练 1840 / 测试 466 张，测试集头尾比 18.6:1），10 组随机种子配对检验：

| 指标 | ResNet-50 基线 | CPGC | 提升 | 配对 t | p 值 |
|---|---|---|---|---|---|
| 准确率 Acc | 89.81 ± 0.84 % | **91.76 ± 0.39 %** | +1.95 pp | 6.74 | 0.0001 |
| Macro-F1 | 82.54 ± 1.84 % | **86.56 ± 1.22 %** | +4.02 pp | 5.78 | 0.0003 |
| Precision | 84.82 ± 2.15 % | 90.84 ± 1.72 % | +6.02 pp | 7.20 | 0.0001 |
| Recall | 81.46 ± 1.88 % | 84.85 ± 1.20 % | +3.39 pp | 4.20 | 0.0023 |

**少数类改善最明显**：折痕（crease，测试集仅 11 张）召回率 53.64% → 74.55%，**+20.91 pp**。

**跨数据集泛化**（X-SDD，冻结骨干 + 仅用 20% 训练数据）：

| 方法 | 准确率 |
|---|---|
| ResNet-50（ImageNet） | 91.42 ± 0.38 % |
| DINO 基线（同骨干） | 92.33 ± 0.25 % |
| **CPGC** | **95.96 ± 0.38 %** |

**几乎零推理开销**：参数量 23.5M → 27.3M（净增 3.72M），平均推理延迟 25.37 ms → 25.32 ms。

完整数据见 [`results/`](results/)。

## 安装

```bash
git clone https://github.com/donglan11/CPGC-dl.git
cd CPGC-dl
pip install -r requirements.txt
```

依赖：Python 3.8+、PyTorch 2.x、torchvision、NumPy、SciPy、scikit-learn。

## 快速开始

### 1. 准备数据

按 ImageFolder 格式放置：

```
data/GC10-DET/
├── train/
│   ├── crease/        # 折痕
│   ├── crescent_gap/  # 月牙缝
│   ├── inclusion/     # 夹杂
│   ├── oil_spot/      # 油斑
│   ├── punching_hole/ # 冲孔
│   ├── rolled_pit/    # 轧坑
│   ├── silk_spot/     # 丝斑
│   ├── waist_folding/ # 腰折
│   ├── water_spot/    # 水斑
│   └── welding_line/  # 焊缝
└── test/
    └── (同样 10 个子目录)
```

GC10-DET 是公开数据集，可从 [lvxiaoming/GC10-DET](https://github.com/lvxiaoming/GC10-DET) 获取。

### 2. 训练

```bash
# 复现论文主表：10 种子 × 30 epoch
python train.py --data-dir data/GC10-DET --epochs 30 --seeds 10

# 单种子快速验证
python train.py --data-dir data/GC10-DET --epochs 30 --seeds 42

# 100 epoch 完整配置
python train.py --data-dir data/GC10-DET --epochs 100 --seeds 10 \
    --baseline-csv results/baseline_10seeds.csv
```

骨干权重（DINO ResNet-50）首次运行自动下载，无需手动放置。

### 3. 结果说明

训练结束后在 `--output-dir` 下得到：

- `cpgc_seed<N>_best.pth` —— 每个种子的最优权重（按测试集最优 epoch 保存）
- `cpgc_results.csv` —— 各种子准确率 + 均值 / 标准差 / 95% CI / 配对 t 检验

## 实验规范

这个仓库的方法论比模型本身更值得参考：

- **多随机种子重复**：所有结论基于 10 组种子（42, 123, 456, 789, 2024, 7, 314, 999, 55, 1337），而非单次运行
- **配对 t 检验**：用 `scipy.stats.ttest_rel` 对同种子结果做配对检验，不是比较两个均值
- **多重比较校正**：逐类分析施加 Bonferroni 与 Benjamini–Hochberg FDR 校正，避免多重检验假阳性
- **区分统计显著与工程显著**：例如测试集仅 466 张，单张图 = 0.21 pp，低于此的变化不作正向声明
- **消融归因**：用留一消融 + 2×2 析因实验定位增益来源，而不是堆叠模块看总提升

`experiments/` 目录下保留了全部消融与对比实验脚本。

## 目录结构

```
CPGC/
├── train.py              # 训练入口（本文方法）
├── requirements.txt
├── LICENSE               # MIT
├── results/              # 论文全部实验数据（CSV）
│   ├── README.md         # 每个文件的口径说明
│   ├── main_10seeds.csv              # 主表：10 种子配对
│   ├── f1_precision_recall.csv       # F1 / Precision / Recall
│   ├── per_class.csv                 # 逐类精度
│   ├── per_class_ttest.csv           # 逐类配对 t 检验
│   ├── xsdd_frozen_20pct.csv         # 跨数据集泛化
│   └── params_latency.csv            # 参数量与推理开销
└── experiments/          # 消融与对比实验
    ├── README.md         # 62 个实验脚本的说明
    ├── core/             # 核心训练变体
    ├── ablation/         # 消融实验
    ├── analysis/         # 敏感性与机制分析
    ├── baselines/        # 基线与对比方法
    ├── visualization/    # Grad-CAM / t-SNE / 训练曲线
    └── stats/            # 统计检验与逐类分析
```

## 主要超参数

| 参数 | 值 | 说明 |
|---|---|---|
| 骨干网络 | DINO ResNet-50 | 自监督预训练，官方检查点 |
| λ_sup | 0.1 | SupCon 损失权重 |
| λ_proto | 0.3 | 原型分类损失权重 |
| SupCon 温度 | 0.07 | |
| 原型温度 | 0.1（可学习） | `τ_proto`，端到端优化 |
| 原型 EMA 动量 m | 0.9 | |
| 原型启用时机 | 第 10 epoch | 待特征结构稳定后再引入 |
| 融合权重 α | σ(θ_α)，初值 0.5 | 可学习标量 |
| 优化器 | Adam，lr 1e-3 | CosineAnnealingLR |
| 批量 / 输入 | 32 / 224×224 | AMP 混合精度 |
| 类别均衡采样器 | 启用 | 所有方法一致，保证公平对比 |

## 已知边界

如实说明这个方法**不解决**什么：

- 数据集规模较小（测试集 466 张），**轧坑（rolled_pit）仅 7 张**，单张图即 14.3 pp，该类结论不稳定
- 与同骨干 DINO+CE 微调基线相比提升仅 +0.86 pp（p=0.076，**未达统计显著**）；方法增益主要体现在**与自监督预训练的协同**（交互项 +1.07 pp, p=0.013）以及冻结骨干场景（+5.58 pp）
- 移除 SupCon 损失对**均值**影响不显著，其价值在于**降低跨种子方差**（std 0.25 → 1.46）
- 尚未在真实产线在线验证

## 论文

方法学部分详见投稿中的论文《原型引导对比学习的钢材表面缺陷长尾分类》。**论文目前处于导师审阅定稿阶段，尚未投稿/发表**，仓库中的实验数据即论文全部数据的来源。

## 引用

```bibtex
@article{cpgc2026,
  title   = {Prototype-Guided Contrastive Learning for Long-Tailed
             Classification of Steel Surface Defects},
  author  = {Wang, Donglan},
  journal = {in preparation},
  year    = {2026}
}
```

## 许可

MIT License，见 [LICENSE](LICENSE)。

依赖的第三方资源：
- [DINO](https://github.com/facebookresearch/dino)（骨干预训练权重，CC BY-NC 4.0）
- [GC10-DET](https://github.com/lvxiaoming/GC10-DET)（数据集，请遵循其原始许可）
