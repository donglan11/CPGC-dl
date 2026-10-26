# -*- coding: utf-8 -*-
"""GC10 冻结版 CPGC 独立核验 + 生成完整汇总 CSV"""
import os, sys, csv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUTPUT_DIR = "D:/EMA/CPGC实验"
sys.path.insert(0, OUTPUT_DIR)
import GC10_冻结版 as g

DATA_DIR = "D:/EMA/GC10_dataset"
SEEDS = [42, 123, 456]
BATCH_SIZE = 32

tf = transforms.Compose([
    transforms.Resize((g.IMG_SIZE, g.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=tf)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  test={len(test_set.samples)}")

def eval_checkpoint(build_fn, tag, is_cpgc):
    results = {}
    for seed in SEEDS:
        model = build_fn(len(test_set.classes)).to(device)
        path = os.path.join(OUTPUT_DIR, f"GC10_frozen_{tag}_seed{seed}_best.pth")
        model.load_state_dict(torch.load(path, map_location=device))
        acc = g.evaluate(model, test_loader, device, is_cpgc)
        results[seed] = acc
        print(f"{tag}-frozen seed={seed}: {acc:.2f}%")
    accs = [results[s] for s in SEEDS]
    mean = sum(accs)/len(accs)
    std = (sum((a-mean)**2 for a in accs)/(len(accs)-1))**0.5
    print(f"{tag} 冻结（GC10）mean={mean:.2f}  std={std:.2f}")
    return results, mean, std

dino_r, dino_m, dino_s = eval_checkpoint(g.build_dino, "DINO", False)
cpgc_r, cpgc_m, cpgc_s = eval_checkpoint(g.build_cpgc, "CPGC", True)

with open(os.path.join(OUTPUT_DIR, "GC10_冻结版结果.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["method"] + [f"seed{s}" for s in SEEDS] + ["mean", "std"])
    w.writerow(["DINO"] + [round(a,2) for a in [dino_r[s] for s in SEEDS]] + [round(dino_m,2), round(dino_s,2)])
    w.writerow(["CPGC"] + [round(a,2) for a in [cpgc_r[s] for s in SEEDS]] + [round(cpgc_m,2), round(cpgc_s,2)])
print("\n已保存: GC10_冻结版结果.csv（完整汇总）")
print(f"配对差 CPGC-DINO: mean={cpgc_m-dino_m:+.2f}  (逐seed: " +
      ", ".join(f"{cpgc_r[s]-dino_r[s]:+.2f}" for s in SEEDS) + ")")
