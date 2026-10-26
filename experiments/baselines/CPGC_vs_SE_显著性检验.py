"""
CPGC vs SE 配对 t 检验
=======================
目的：验证 CPGC 是否在统计意义上显著优于 SE 注意力基线
数据：3 种子（42, 123, 456），直接读已有 CSV
"""
import os, csv
import numpy as np
from scipy import stats

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== 读数据 =====
def read_csv(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {int(r["seed"]): float(r["best_acc"])
                for r in reader if r["seed"] and r["seed"].isdigit()}

cpgc_path = os.path.join(OUTPUT_DIR, "CPGC_3种子结果.csv")
se_path = os.path.join(OUTPUT_DIR, "SE_3种子结果.csv")

cpgc = read_csv(cpgc_path)
se = read_csv(se_path)

SEEDS = sorted(cpgc.keys())  # [42, 123, 456]

# ===== 配对 t 检验 =====
cpgc_vals = [cpgc[s] for s in SEEDS]
se_vals = [se[s] for s in SEEDS]
diffs = [c - s for c, s in zip(cpgc_vals, se_vals)]

t_stat, p_val = stats.ttest_rel(cpgc_vals, se_vals)
mean_diff = np.mean(diffs)
ci95 = stats.t.interval(0.95, df=len(diffs) - 1, loc=mean_diff,
                         scale=stats.sem(diffs))

# ===== 输出 =====
print("=" * 50)
print("CPGC vs SE 配对 t 检验")
print("=" * 50)
print(f"{'Seed':>6}  {'SE':>8}  {'CPGC':>8}  {'Diff':>8}")
print("-" * 34)
for i, s in enumerate(SEEDS):
    print(f"{s:>6}  {se_vals[i]:>8.2f}  {cpgc_vals[i]:>8.2f}  {diffs[i]:>+8.2f}")
print("-" * 34)
print(f"{'Mean':>6}  {np.mean(se_vals):>8.2f}  {np.mean(cpgc_vals):>8.2f}"
      f"  {mean_diff:>+8.2f}")
print(f"\n配对 t = {t_stat:.3f}")
print(f"自由度 df = {len(SEEDS) - 1}")
print(f"p 值 (双尾) = {p_val:.4f}")
ci_low, ci_high = ci95
print(f"差值 95% CI = [{ci_low:.2f}, {ci_high:.2f}]")
print(f"均值差值 = {mean_diff:.2f}")
print()
if p_val < 0.05:
    print("结论: p < 0.05，CPGC 在 3 种子尺度下显著优于 SE")
else:
    print("结论: p >= 0.05，3 种子尺度下未达统计显著")
print("=" * 50)

# ===== 保存 CSV =====
csv_path = os.path.join(OUTPUT_DIR, "CPGC_vs_SE_显著性检验.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["seed", "se_acc", "cpgc_acc", "diff"])
    for i, s in enumerate(SEEDS):
        w.writerow([s, se_vals[i], cpgc_vals[i], round(diffs[i], 2)])
    w.writerow([])
    w.writerow(["paired_t", round(t_stat, 3)])
    w.writerow(["df", len(SEEDS) - 1])
    w.writerow(["p_value", round(p_val, 4)])
    w.writerow(["diff_mean", round(mean_diff, 2)])
    w.writerow(["diff_95CI_lower", round(ci_low, 2)])
    w.writerow(["diff_95CI_upper", round(ci_high, 2)])
    w.writerow(["conclusion",
                "significant" if p_val < 0.05 else "not_significant"])
print(f"已保存: {csv_path}")