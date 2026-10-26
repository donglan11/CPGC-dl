# -*- coding: utf-8 -*-
"""生成 CPGC_BS_3种子训练.py：复制 CPGC_3种子训练.py 并做精准修改。"""
src = r"D:\EMA\CPGC实验\CPGC_3种子训练.py"
dst = r"D:\EMA\CPGC实验\CPGC_BS_3种子训练.py"

c = open(src, encoding="utf-8").read()
orig_len = len(c)

def rep(old, new, tag, count=1):
    global c
    assert old in c, f"[FAIL] 锚点未命中: {tag}"
    c = c.replace(old, new, count)
    print(f"[OK] {tag}")

# 0) 头部 docstring 整体替换（从第一个三引号到其闭合）
doc_start = c.index('"""')
doc_end = c.index('"""', doc_start + 3)
new_doc = ('"""CPGC + Balanced Softmax 对照实验 (3 种子 x 2 配置)\n'
           '============================================================\n'
           '目的：验证"长尾先验校正"(Balanced Softmax) 能否叠加到 CPGC（正交增益）。\n'
           '改动：融合分类头的 CE 换成 Balanced Softmax（logits += log(prior)），\n'
           '      原型辅助损失保持 CE 不变；训练与推理同规则（BS 论文约定）。\n'
           '配置：no_sampler（BS 文献标准用法）/ with_sampler（与主结果协议对齐）。\n'
           '注意：主结果 CPGC 开 WeightedRandomSampler，叠加 BS 属双重校正，预期掉点；\n'
           '      no_sampler 版是 BS 最优配置。两个结果分别对论文叙事有价值。\n'
           '"""')
c = c[:doc_start] + new_doc + c[doc_end + 3:]  # 跳过原闭合三引号
print("[OK] docstring（整块替换）")

# 1) __init__ 加 class_prior 参数
rep("    def __init__(self, num_classes=10, proj_dim=128):",
    "    def __init__(self, num_classes=10, proj_dim=128, class_prior=None):",
    "init签名")

# 2) 注册 log_prior buffer（原型温度之后）
rep("        # 原型温度 (可学习)\n        self.log_proto_temp = nn.Parameter(torch.tensor(math.log(0.1)))",
    "        # 原型温度 (可学习)\n        self.log_proto_temp = nn.Parameter(torch.tensor(math.log(0.1)))\n\n"
    "        # Balanced Softmax 类别先验 log(pi)；class_prior=None 时不用 BS\n"
    "        if class_prior is not None:\n"
    '            self.register_buffer("log_prior", torch.log(class_prior))\n'
    "            self.use_bs = True\n"
    "        else:\n"
    "            self.log_prior = None\n"
    "            self.use_bs = False",
    "log_prior buffer")

# 3) forward 末尾加 BS 先验
rep("        return proj, mlp_logits, proto_scores, final_logits",
    "        if self.use_bs:\n"
    "            final_logits = final_logits + self.log_prior\n\n"
    "        return proj, mlp_logits, proto_scores, final_logits",
    "forward BS先验")

# 4) train_one 签名 + 模型构造 + 输出路径
rep("def train_one(seed, train_loader, test_loader, num_classes, device):",
    "def train_one(seed, train_loader, test_loader, num_classes, device, tag):",
    "train_one签名")
rep("    model = CPGC(num_classes=num_classes).to(device)",
    "    model = CPGC(num_classes=num_classes, class_prior=CLASS_PRIOR).to(device)",
    "模型构造")
rep('    model_path = os.path.join(OUTPUT_DIR, f"CPGC_seed{seed}_best.pth")',
    '    model_path = os.path.join(OUTPUT_DIR, f"CPGC_BS_{tag}_seed{seed}_best.pth")',
    "输出路径")

# 5) CONFIGS 常量
rep("EPOCHS = 100\nBATCH_SIZE = 32",
    "EPOCHS = 100\nBATCH_SIZE = 32\n"
    "# 两个配置：no_sampler（BS 文献标准用法）/ with_sampler（与主结果协议对齐）\n"
    "CONFIGS = [\n"
    '    {"use_sampler": False, "tag": "no_sampler"},\n'
    '    {"use_sampler": True,  "tag": "with_sampler"},\n'
    "]",
    "CONFIGS")

# 6) main：train_loader 构建 → 移入循环；补 CLASS_PRIOR
rep("    if BALANCE_CLASSES:\n"
    "        sample_weights = [1.0 / class_counts[label] for _, label in train_set.samples]\n"
    "        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)\n"
    "        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)\n"
    "    else:\n"
    "        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)\n"
    "    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)",
    "    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)\n\n"
    "    # Balanced Softmax 类别先验（训练集自然分布）\n"
    "    global CLASS_PRIOR\n"
    "    total = sum(class_counts)\n"
    "    CLASS_PRIOR = torch.tensor([c / total for c in class_counts], dtype=torch.float32)\n"
    '    print(f"Class prior: {CLASS_PRIOR.tolist()}")',
    "main loader/prior")

# 7) main：results 循环 + CSV → CONFIGS 循环版
rep("    results = {}\n"
    "    for seed in SEEDS:\n"
    '        print(f"\\n--- seed={seed} ---")\n'
    "        t_start = time.time()\n"
    "        res = train_one(seed, train_loader, test_loader, num_classes, device)\n"
    "        results[seed] = res\n"
    '        print(f"<<< CPGC seed={seed} done. Best acc {res[\'best_acc\']:.2f}%  ({(time.time()-t_start)/60:.1f} min)")\n\n'
    "    accs = [results[s][\"best_acc\"] for s in SEEDS]\n"
    '    print(f"\\n{\'=\'*50}")\n'
    '    print(f"CPGC (3 种子)")\n'
    '    print(f"{\'=\'*50}")\n'
    "    for s in SEEDS:\n"
    '        print(f"  seed={s}: {results[s][\'best_acc\']:.2f}%")\n'
    '    print(f"  mean={np.mean(accs):.2f}%  std={np.std(accs, ddof=1):.2f}%")\n'
    '    print(f"  vs baseline 89.81%: diff={np.mean(accs)-89.81:+.2f}%")\n'
    '    print(f"  vs DINO+SupCon+MLP 91.34%: diff={np.mean(accs)-91.34:+.2f}%")\n\n'
    '    csv_path = os.path.join(OUTPUT_DIR, "CPGC_3种子结果.csv")\n'
    '    with open(csv_path, "w", newline="", encoding="utf-8") as f:\n'
    "        w = csv.writer(f)\n"
    '        w.writerow(["seed", "best_acc"])\n'
    "        for s in SEEDS:\n"
    '            w.writerow([s, round(results[s]["best_acc"], 2)])\n'
    "        w.writerow([])\n"
    '        w.writerow(["mean", round(np.mean(accs), 2)])\n'
    '        w.writerow(["std", round(np.std(accs, ddof=1), 2)])\n'
    '        w.writerow(["vs_baseline", round(np.mean(accs) - 89.81, 2)])\n'
    '        w.writerow(["vs_dino_supcon_mlp", round(np.mean(accs) - 91.34, 2)])\n'
    '    print(f"\\n已保存: {csv_path}")',
    "    all_rows = []\n"
    "    for cfg in CONFIGS:\n"
    '        tag = cfg["tag"]\n'
    '        use_sampler = cfg["use_sampler"]\n'
    '        print(f"\\n{\'=\'*60}\\n配置: CPGC + Balanced Softmax, use_sampler={use_sampler}\\n{\'=\'*60}")\n'
    "        if use_sampler:\n"
    "            sample_weights = [1.0 / class_counts[label] for _, label in train_set.samples]\n"
    "            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)\n"
    "            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)\n"
    "        else:\n"
    "            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)\n\n"
    "        results = {}\n"
    "        for seed in SEEDS:\n"
    '            print(f"\\n--- seed={seed} ---")\n'
    "            t_start = time.time()\n"
    "            res = train_one(seed, train_loader, test_loader, num_classes, device, tag)\n"
    "            results[seed] = res\n"
    '            print(f"<<< CPGC BS {tag} seed={seed} done. Best acc {res[\'best_acc\']:.2f}%  ({(time.time()-t_start)/60:.1f} min)")\n\n'
    "        accs = [results[s][\"best_acc\"] for s in SEEDS]\n"
    "        mean = float(np.mean(accs))\n"
    "        std = float(np.std(accs, ddof=1))\n"
    "        all_rows.append([tag, round(mean, 2), round(std, 2)] + [round(results[s][\"best_acc\"], 2) for s in SEEDS])\n"
    '        print(f"  {tag}: mean={mean:.2f}%  std={std:.2f}%")\n\n'
    '    csv_path = os.path.join(OUTPUT_DIR, "CPGC_BS_3种子结果.csv")\n'
    '    with open(csv_path, "w", newline="", encoding="utf-8") as f:\n'
    "        w = csv.writer(f)\n"
    '        w.writerow(["config", "mean", "std", "seed42", "seed123", "seed456"])\n'
    "        for row in all_rows:\n"
    "            w.writerow(row)\n"
    '    print(f"\\n已保存: {csv_path}")',
    "main 循环/CSV")

open(dst, "w", encoding="utf-8", newline="\n").write(c)
print(f"\n生成完毕: {dst}  ({(len(c)-orig_len)/1024:.1f} KB 增量)")
