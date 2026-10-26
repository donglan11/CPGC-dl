"""
逐类精度分析 —— 逐种子导出补跑版（不改动 逐类分析_10种子.py）
================================================================
目的：千问要求为 inclusion 回退跑配对 t 检验，但原脚本只落盘 mean/std，
      逐种子矩阵未保存。本脚本复用同一推理逻辑，额外把「每类每种子」的
      baseline / CPGC 精度导出为 CSV，供 scipy.stats.ttest_rel 使用。

复用：直接 import 原脚本的 build_loader / per_class_acc，确保推理逻辑零偏差。
输出：CPGC实验/逐类分析_10种子_逐种子明细.csv
      （长表：seed, class, baseline_acc, cpgc_acc, diff）
"""
import os, sys, csv
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50

CPGC_DIR = os.path.join("D:/EMA", "CPGC实验")
sys.path.insert(0, CPGC_DIR)
sys.path.insert(0, "D:/EMA")

# 复用原脚本的推理函数（保证与已发布 CSV 同源同逻辑）
import importlib
_orig = importlib.import_module("逐类分析_10种子")
build_loader = _orig.build_loader
per_class_acc = _orig.per_class_acc
CPGC = _orig.CPGC
SEEDS = _orig.SEEDS
NUM_CLASSES = _orig.NUM_CLASSES
device = _orig.device
BASE_DIR = _orig.BASE_DIR

if __name__ == "__main__":
    loader, class_names = build_loader()
    print(f"Device: {device}  Test: {len(loader.dataset)} images  Classes: {len(class_names)}")

    rows = []          # (seed, class, base_acc, cpgc_acc, diff)
    base_cls, cpgc_cls, base_all, cpgc_all = [], [], [], []

    for seed in SEEDS:
        # ---- 基线 ----
        bp = os.path.join(BASE_DIR, f"ResNet50_seed{seed}_best.pth")
        bm = resnet50(weights=None)
        bm.fc = nn.Linear(bm.fc.in_features, NUM_CLASSES)
        bm.load_state_dict(torch.load(bp, map_location=device))
        bm = bm.to(device).eval()
        bc, bo = per_class_acc(bm, loader, False)
        del bm
        torch.cuda.empty_cache()

        # ---- CPGC ----
        cp = os.path.join(CPGC_DIR, f"CPGC_10seed_seed{seed}_best.pth")
        cm = CPGC(num_classes=NUM_CLASSES).to(device)
        cm.load_state_dict(torch.load(cp, map_location=device))
        cm.eval()
        cc, co = per_class_acc(cm, loader, True)
        del cm
        torch.cuda.empty_cache()

        base_cls.append(bc); cpgc_cls.append(cc)
        base_all.append(bo); cpgc_all.append(co)

        for i, c in enumerate(class_names):
            rows.append([seed, c, round(float(bc[i]), 4), round(float(cc[i]), 4), round(float(cc[i]-bc[i]), 4)])

        print(f"seed={seed:<5} baseline {bo:.2f}%   CPGC {co:.2f}%   diff {co-bo:+.2f}")

    out = os.path.join(CPGC_DIR, "逐类分析_10种子_逐种子明细.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "class", "baseline_acc", "cpgc_acc", "diff"])
        w.writerows(rows)
    print(f"\nSaved: {out}")

    # ===== 顺带跑一遍每类的配对 t 检验（千问要求的核心交付）=====
    from scipy import stats
    base_arr = np.array(base_cls)   # (10, 10)
    cpgc_arr = np.array(cpgc_cls)
    print("\n" + "=" * 88)
    print(f"{'Class':<18}{'Base':>10}{'CPGC':>10}{'Diff':>9}{'t':>9}{'p':>10}{'显著(p<.05)':>12}")
    print("-" * 88)
    report = []
    for i, c in enumerate(class_names):
        b, p_ = base_arr[:, i], cpgc_arr[:, i]
        t, pv = stats.ttest_rel(p_, b)
        sig = "是" if pv < 0.05 else "否"
        print(f"{c:<18}{b.mean():>10.2f}{p_.mean():>10.2f}{p_.mean()-b.mean():>+9.2f}{t:>9.3f}{pv:>10.4f}{sig:>12}")
        report.append((c, b.mean(), p_.mean(), p_.mean()-b.mean(), t, pv, sig))
    print("=" * 88)

    # 总体
    t_all, p_all = stats.ttest_rel(np.array(cpgc_all), np.array(base_all))
    print(f"{'Overall':<18}{np.mean(base_all):>10.2f}{np.mean(cpgc_all):>10.2f}"
          f"{np.mean(cpgc_all)-np.mean(base_all):>+9.2f}{t_all:>9.3f}{p_all:>10.4f}"
          f"{'是' if p_all<0.05 else '否':>12}")

    # 导出 t 检验结果
    out2 = os.path.join(CPGC_DIR, "逐类配对t检验_10种子.csv")
    with open(out2, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "baseline_mean", "cpgc_mean", "diff", "paired_t", "p_value", "significant_0.05"])
        for r in report:
            w.writerow([r[0], round(r[1],2), round(r[2],2), round(r[3],2), round(r[4],4), round(r[5],4), r[6]])
    print(f"Saved: {out2}")
