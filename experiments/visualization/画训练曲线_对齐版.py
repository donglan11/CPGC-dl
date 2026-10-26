"""
训练曲线对比图（对齐版，seed=42）
=============================================
为什么需要这一版：
  旧图 `训练曲线_对比图.png` 的基线来自 `Baseline_training_log_seed42.csv`
  （EPOCHS=100 + ImageNet **V2** 权重，终点 best=91.2），
  与表 1 的 30 epoch / V1 权重基线（90.27）不是同一配置 → 图与表口径冲突。

本脚本改用**同脚本只改 EPOCHS 30→100**的对齐基线日志：
  ResNet50基线实验/train_100ep.log  （V1 权重，seed=42，best=90.56@Ep83）
CPGC 侧沿用 `CPGC_training_log_seed42.csv`（best=91.85，与可复现权重一致）。

注意：对齐基线脚本当时未记录 loss，故本图只画准确率三项（Train / Test / Best），
不画 loss —— 宁可少一个面板，也不要拿不可比的 loss 曲线凑数。
"""
import csv
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BASELINE_LOG = os.path.join(ROOT, "ResNet50基线实验", "train_100ep.log")
CPGC_LOG = os.path.join(HERE, "CPGC_training_log_seed42.csv")
OUT_PNG = os.path.join(HERE, "训练曲线_对齐版.png")

SEED = 42
BASELINE_30EP_MEAN = 90.27   # 表 1 基线（3 种子配对，30 epoch / V1）
BASELINE_30EP_SEED42 = 90.56  # 同名脚本 30 epoch 口径下 seed=42 的值（ResNet50_基线结果.csv）
CPGC_BEST = 91.85            # 可复现权重 CPGC_10seed_seed42_best.pth 实测

LINE_RE = re.compile(
    r"\[ResNet50-100ep seed=(\d+)\]\s+Ep\s+(\d+)/\d+\s+"
    r"Train\s+([\d.]+)%\s+Test\s+([\d.]+)%\s+F1\s+([\d.]+)%\s+Best\s+([\d.]+)%"
)


def read_aligned_baseline(seed=SEED):
    """从训练日志解析出逐 epoch 的 train_acc / test_acc / best_acc。"""
    ep, ta, tea, ba = [], [], [], []
    with open(BASELINE_LOG, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            if int(m.group(1)) != seed:
                continue
            ep.append(int(m.group(2)))
            ta.append(float(m.group(3)))
            tea.append(float(m.group(4)))
            ba.append(float(m.group(6)))
    if not ep:
        raise RuntimeError(f"未能从 {BASELINE_LOG} 解析出 seed={seed} 的逐 epoch 记录")
    return ep, ta, tea, ba


def read_cpgc():
    ep, ta, tea, ba = [], [], [], []
    with open(CPGC_LOG, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            ta.append(float(row["train_acc"]))
            tea.append(float(row["test_acc"]))
            ba.append(float(row["best_acc"]))
    return ep, ta, tea, ba


b_ep, b_ta, b_tea, b_ba = read_aligned_baseline()
c_ep, c_ta, c_tea, c_ba = read_cpgc()

CB, CC = "#1f77b4", "#d62728"
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
fig.suptitle(
    f"Training Curves — Aligned Budget (ResNet-50 baseline, seed={SEED}, 100 epochs)",
    fontsize=13, fontweight="bold",
)

panels = [
    (axes[0], b_ta, c_ta, "Train Accuracy", "Training Accuracy (%)"),
    (axes[1], b_tea, c_tea, "Test Accuracy", "Test Accuracy (%)"),
    (axes[2], b_ba, c_ba, "Best Accuracy", "Best Accuracy (%)"),
]

for ax, b, c, title, ylabel in panels:
    ax.plot(b_ep, b, label="ResNet-50 Baseline (100 ep)", color=CB, linewidth=1.5)
    ax.plot(c_ep, c, label="CPGC (Ours, 100 ep)", color=CC, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 101)
    ax.grid(True, alpha=0.3)

axes[2].axhline(BASELINE_30EP_MEAN, color="#7f7f7f", ls=":", lw=1.2)
axes[2].text(2, BASELINE_30EP_MEAN + 0.35, f"Table-1 baseline, 30 ep = {BASELINE_30EP_MEAN:.2f}%",
             fontsize=8, color="#5f5f5e", va="bottom")
axes[2].legend(loc="lower right", fontsize=8.5)

b_final, c_final = b_ba[-1], c_ba[-1]
axes[2].annotate(f"{b_final:.2f}%", xy=(b_ep[-1], b_final), xytext=(-46, -22),
                 textcoords="offset points", fontsize=9, color=CB,
                 arrowprops=dict(arrowstyle="-", color=CB, lw=0.8))
axes[2].annotate(f"{c_final:.2f}%", xy=(c_ep[-1], c_final), xytext=(-30, 10),
                 textcoords="offset points", fontsize=9, color=CC,
                 arrowprops=dict(arrowstyle="-", color=CC, lw=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()

print(f"已保存: {OUT_PNG}")
b_best_ep = b_ep[b_ba.index(max(b_ba))]
print(f"\n对齐口径（seed={SEED}, 100 epoch, ImageNet V1）")
print(f"  Baseline 100 ep best = {b_final:.2f}%   (@Ep{b_best_ep})")
print(f"  CPGC     100 ep best = {c_final:.2f}%")
print(f"  差            = {c_final - b_final:+.2f}")
print(f"  参考：表1 30ep 3种子基线均值 = {BASELINE_30EP_MEAN:.2f}%")
print(f"  → 同种子配对（seed42）：30ep {BASELINE_30EP_SEED42:.2f}% vs 100ep {b_final:.2f}%"
      f" = {b_final - BASELINE_30EP_SEED42:+.2f}（epoch 几乎不影响基线）")
