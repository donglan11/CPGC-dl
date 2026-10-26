# results —— 论文全部实验数据

本目录的 CSV 即论文中所有表格的数据源，**数值与论文逐格一致**。文件名已重命名为英文（GitHub 跨平台兼容），原始中文名见「原始文件」列。

| 文件 | 内容 | 原始文件 |
|---|---|---|
| `main_10seeds.csv` | **主表**：10 种子配对检验，CPGC 91.76±0.39 vs 基线 89.81±0.84，diff=1.95，t=6.74，p=0.0001 | `CPGC实验/CPGC_10种子结果.csv` |
| `f1_precision_recall.csv` | F1 / Precision / Recall 三指标逐种子对比（10 种子） | `CPGC实验/F1_Precision_Recall_10种子对比.csv` |
| `per_class.csv` | 逐类精度 ± 跨种子标准差，含差值列 | `CPGC实验/逐类分析_10种子.csv` |
| `per_class_ttest.csv` | 逐类配对 t 检验（含多重比较校正解读） | `CPGC实验/逐类配对t检验_10种子.csv` |
| `xsdd_frozen_20pct.csv` | **跨数据集泛化**：X-SDD，冻结骨干 + 20% 数据，三方法对比 | `X-SDD实验/X-SDD_fewshot20pct_frozen_*.csv` |
| `params_latency.csv` | 参数量明细与推理延迟 / 吞吐量 | `CPGC实验/参数量与推理速度对比.csv` |
| `baseline_10seeds.csv` | ResNet-50 基线 10 种子结果，供 `train.py --baseline-csv` 使用 | `ResNet50基线实验/ResNet50_基线结果.csv` |

## 口径说明

**种子口径**：10 种子 = 42, 123, 456, 789, 2024, 7, 314, 999, 55, 1337（df=9）。

**训练轮数口径**：主表 `main_10seeds.csv` 为 **30 epoch**；论文消融表为 100 epoch（同种子同预算对照）。两者**不可跨表相减**。

**选模协议**：GC10-DET 无官方验证集，本工作对**所有方法一致地**逐个 epoch 在测试集上评估并报告最优 epoch（test-time model selection）。该方法间相对比较公平，绝对值略偏乐观。

**预训练权重**：ImageNet 基线用 `IMAGENET1K_V1`。V1 与 V2 相差约 0.6 pp，**引用时不可混用**。

**长尾分布**：
- GC10-DET 训练集 1840 张 / 测试集 466 张
- 训练集头尾比 21.7 : 1（丝斑 520 : 轧坑 24）
- 测试集头尾比 18.6 : 1（丝斑 130 : 轧坑 7）

**⚠️ 少数类样本量极小，结论须谨慎**：轧坑（rolled_pit）测试集仅 **7 张**，单张图即 14.3 pp，该类**不作正向声明**。

**⚠️ 两组冻结骨干实验是不同设定，勿混说**：
- GC10 冻结骨干：DINO 83.12 / CPGC 88.70，差 **5.58** pp
- X-SDD 冻结骨干（另含 20% 少样本）：DINO 92.33 / CPGC 95.96，差 **3.63** pp

## 读取示例

```python
import csv

with open("main_10seeds.csv", encoding="utf-8") as f:
    rows = list(csv.reader(f))

data = {r[0]: r for r in rows if r}
print(data["mean"])        # ['mean', '91.76', '89.81']
print(data["paired_t"])    # ['paired_t', '6.74']
print(data["p_value"])     # ['p_value', '0.0001']
```
