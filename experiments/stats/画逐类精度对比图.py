"""
逐类精度对比图 — CPGC vs Baseline（10 种子 mean±std）
======================================================
数据源：`逐类分析_10种子.csv`（由 逐类分析_10种子.py 用 10 个可复现种子产出）

产出：`逐类精度_对比图.png`
  (a) 分组条形图：每类 Baseline / CPGC 精度，误差棒 = 跨 10 种子标准差
  (b) 差值条形图：CPGC − Baseline，正负分色，突出长尾类

设计说明（写给论文）：
  - 类别顺序按测试集样本数升序、长尾类置于**顶部**，y 轴标签带样本数，便于判断统计功效
  - rolled_pit 仅 7 张测试图，单图 = 14.3 个百分点 → 必须看误差棒，不可只看均值
  - CPGC 的主要收益集中在 crease(+20.91) 与 waist_folding(+7.00) 两个长尾类
  - cpgc_std < baseline_std 的 5 个类（含 rolled_pit 14.21→10.54）说明 CPGC 还降低了跨种子方差

用法：D:/Anaconda/envs/my3080/python.exe 画逐类精度对比图.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "逐类分析_10种子.csv")
TEST_DIR = os.path.join(os.path.dirname(HERE), "GC10_dataset", "test")
OUT_PATH = os.path.join(HERE, "逐类精度_对比图.png")

C_BASE = "#1f77b4"      # Baseline 蓝
C_CPGC = "#d62728"      # CPGC 红
C_POS = "#d62728"       # 提升
C_NEG = "#7f8fa6"       # 下降


def count_test_samples(test_dir):
    """统计每个类别的测试集图片数"""
    counts = {}
    if not os.path.isdir(test_dir):
        return counts
    for name in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, name)
        if os.path.isdir(d):
            counts[name] = len([f for f in os.listdir(d)
                                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    return counts


def load_csv(path):
    rows = []
    overall = None
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = {
                "class": r["class"],
                "b_mean": float(r["baseline_mean"]), "b_std": float(r["baseline_std"]),
                "c_mean": float(r["cpgc_mean"]), "c_std": float(r["cpgc_std"]),
                "diff": float(r["diff"]),
            }
            (rows.append(item) if r["class"] != "overall" else None)
            if r["class"] == "overall":
                overall = item
    return rows, overall


rows, overall = load_csv(CSV_PATH)
counts = count_test_samples(TEST_DIR)

# 按测试集样本数升序排列（长尾在左）；无计数时退化为按提升量降序
order = sorted(range(len(rows)),
               key=lambda i: (counts.get(rows[i]["class"], 10 ** 6), -rows[i]["diff"]))
rows = [rows[i] for i in order]

names = [f"{r['class']}\n(n={counts.get(r['class'], '?')})" for r in rows]
b_mean = np.array([r["b_mean"] for r in rows])
b_std = np.array([r["b_std"] for r in rows])
c_mean = np.array([r["c_mean"] for r in rows])
c_std = np.array([r["c_std"] for r in rows])
diff = np.array([r["diff"] for r in rows])

y = np.arange(len(rows))
h = 0.38

fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                         gridspec_kw={"width_ratios": [1.35, 1]})

# ---------- (a) 分组条形图 ----------
ax = axes[0]
ax.barh(y + h / 2, b_mean, height=h, xerr=b_std, color=C_BASE, alpha=0.9,
        label="ResNet-50 Baseline", error_kw=dict(ecolor="#333", lw=1, capsize=3))
ax.barh(y - h / 2, c_mean, height=h, xerr=c_std, color=C_CPGC, alpha=0.9,
        label="CPGC (Ours)", error_kw=dict(ecolor="#333", lw=1, capsize=3))
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlim(0, 112)
ax.set_xlabel("Per-class Accuracy (%)", fontsize=11)
ax.set_title("(a) Per-class accuracy, mean ± std over 10 seeds", fontsize=12)
ax.axvline(89.81, color="#888", ls=":", lw=1, label="baseline overall 89.81%")
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()   # 样本数少的类放顶部，长尾优先被看到

# ---------- (b) 差值条形图 ----------
ax = axes[1]
colors = [C_POS if d > 0 else C_NEG for d in diff]
ax.barh(y, diff, height=0.6, color=colors, alpha=0.9)
ax.axvline(0, color="#333", lw=1)
for i, d in enumerate(diff):
    off = 0.6 if d >= 0 else -0.6
    ax.text(d + off, i, f"{d:+.2f}", va="center",
            ha="left" if d >= 0 else "right", fontsize=8.5,
            color="#b02020" if d > 0 else "#555")
ax.set_yticks(y)
ax.set_yticklabels([r["class"] for r in rows], fontsize=9)
ax.set_xlim(-8, 25)
ax.set_xlabel("Accuracy change (%)", fontsize=11)
ax.set_title(f"(b) CPGC − Baseline  (overall {overall['diff']:+.2f})", fontsize=12)
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()

fig.suptitle("Per-class Analysis on GC10-DET Test Set (466 images, seeds: "
             "42/123/456/789/2024/7/314/999/55/1337)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

# ---------- 控制台复核 ----------
print(f"\n{'Class':<18}{'n':>4}{'Baseline':>16}{'CPGC':>16}{'Diff':>9}")
print("-" * 64)
for r, n in zip(rows, [counts.get(r["class"], 0) for r in rows]):
    print(f"{r['class']:<18}{n:>4}{r['b_mean']:>8.2f} ± {r['b_std']:<4.2f}"
          f"{r['c_mean']:>8.2f} ± {r['c_std']:<4.2f}{r['diff']:>+9.2f}")
print("-" * 64)
print(f"{'Overall':<18}{sum(counts.values()):>4}"
      f"{overall['b_mean']:>8.2f} ± {overall['b_std']:<4.2f}"
      f"{overall['c_mean']:>8.2f} ± {overall['c_std']:<4.2f}{overall['diff']:>+9.2f}")
print(f"\nstd 下降的类（CPGC 更稳）：", end="")
print(", ".join(f"{r['class']} {r['b_std']:.2f}→{r['c_std']:.2f}"
                for r in rows if r["c_std"] < r["b_std"]))
