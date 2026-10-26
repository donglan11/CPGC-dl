# -*- coding: utf-8 -*-
"""从 CPGC_BS_no_sampler_10种子补跑.py 生成 CPGC_LDAM_3种子.py
把 Balanced Softmax(log_prior) 换成 LDAM margin = C / n_c^(1/4)
"""
import io

SRC = "CPGC_BS_no_sampler_10种子补跑.py"
DST = "CPGC_LDAM_3种子.py"

with io.open(SRC, "r", encoding="utf-8") as f:
    s = f.read()

pairs = [
    # 1. docstring
    (
'''"""CPGC + Balanced Softmax 补跑 (num_workers=0, 修复 error1455)
============================================================
目的：验证"长尾先验校正"(Balanced Softmax) 能否叠加到 CPGC（正交增益）。
改动：融合分类头的 CE 换成 Balanced Softmax（logits += log(prior)），
      原型辅助损失保持 CE 不变；训练与推理同规则（BS 论文约定）。
配置：no_sampler（BS 文献标准用法）/ with_sampler（与主结果协议对齐）。
注意：主结果 CPGC 开 WeightedRandomSampler，叠加 BS 属双重校正，预期掉点；
      no_sampler 版是 BS 最优配置。两个结果分别对论文叙事有价值。
"""''',
'''"""CPGC + LDAM（Label-Distribution-Aware Margin）外部长尾对照 (num_workers=0)
============================================================
目的：补表1-c 外部长尾标准方法对照（BS 变"自己人"后表1-c 缺外部对照）。
LDAM (Cao et al. NeurIPS 2019)：logits 减去类别感知 margin = C / n_c^(1/4)，
      样本越少的类 margin 越大，拉开稀有类决策边界。
配置：no_sampler（LDAM 文献标准用法，无重采样）。
注意：这是外部对照方法，与 CPGC/BS 区分开；3 种子，与 SE/CBAM/SimAM 同规。
"""'''
    ),
    # 2. 加 LDAM_C 常量
    (
        "PROTO_START = 10     # 前 10 个 epoch 不做原型分类（特征不稳定）",
        "PROTO_START = 10     # 前 10 个 epoch 不做原型分类（特征不稳定）\n"
        "LDAM_C = 1.0         # LDAM margin 缩放系数（原文通过验证集选，此处取 1.0）"
    ),
    # 3. __init__ 签名
    (
        "    def __init__(self, num_classes=10, proj_dim=128, class_prior=None):",
        "    def __init__(self, num_classes=10, proj_dim=128, class_counts=None):"
    ),
    # 4. log_prior -> ldam_margin
    (
'''        # Balanced Softmax 类别先验 log(pi)；class_prior=None 时不用 BS
        if class_prior is not None:
            self.register_buffer("log_prior", torch.log(class_prior))
            self.use_bs = True
        else:
            self.log_prior = None
            self.use_bs = False''',
'''        # LDAM 类别感知 margin C/n_c^(1/4)；class_counts=None 时不用 LDAM
        if class_counts is not None:
            self.register_buffer("ldam_margin", LDAM_C / torch.pow(class_counts.float(), 0.25))
            self.use_ldam = True
        else:
            self.ldam_margin = None
            self.use_ldam = False'''
    ),
    # 5. forward 里应用
    (
'''        if self.use_bs:
            final_logits = final_logits + self.log_prior''',
'''        if self.use_ldam:
            final_logits = final_logits - self.ldam_margin'''
    ),
    # 6. 模型实例化
    (
        "    model = CPGC(num_classes=num_classes, class_prior=CLASS_PRIOR).to(device)",
        "    model = CPGC(num_classes=num_classes, class_counts=CLASS_COUNTS).to(device)"
    ),
    # 7. 权重文件名
    (
        '    model_path = os.path.join(OUTPUT_DIR, f"CPGC_BS_{tag}_seed{seed}_best.pth")',
        '    model_path = os.path.join(OUTPUT_DIR, f"CPGC_LDAM_{tag}_seed{seed}_best.pth")'
    ),
    # 8. CLASS_PRIOR -> CLASS_COUNTS
    (
'''    # Balanced Softmax 类别先验（训练集自然分布）
    global CLASS_PRIOR
    total = sum(class_counts)
    CLASS_PRIOR = torch.tensor([c / total for c in class_counts], dtype=torch.float32)
    print(f"Class prior: {CLASS_PRIOR.tolist()}")''',
'''    # LDAM 类别感知 margin（训练集自然分布）
    global CLASS_COUNTS
    CLASS_COUNTS = torch.tensor(class_counts, dtype=torch.float32)
    print(f"Class counts: {CLASS_COUNTS.tolist()}")'''
    ),
    # 9. RUN_QUEUE
    (
'''    # 补跑队列：no_sampler seed456（上次 error1455 崩溃）+ with_sampler 3 种子（未跑）
    RUN_QUEUE = [
        (False, "no_sampler", [789, 2024, 7, 314, 999, 55, 1337]),
    ]''',
'''    # LDAM 外部对照：no_sampler 3 种子（与 SE/CBAM/SimAM 同规）
    RUN_QUEUE = [
        (False, "no_sampler", [42, 123, 456]),
    ]'''
    ),
    # 10. 配置打印
    (
        '        print(f"\\n{\'=\'*60}\\n配置: CPGC + Balanced Softmax, use_sampler={use_sampler}\\n{\'=\'*60}")',
        '        print(f"\\n{\'=\'*60}\\n配置: CPGC + LDAM, use_sampler={use_sampler}\\n{\'=\'*60}")'
    ),
    # 11. done 打印
    (
        '            print(f"<<< CPGC BS {tag} seed={seed} done. Best acc {res[\'best_acc\']:.2f}%  ({(time.time()-t_start)/60:.1f} min)")',
        '            print(f"<<< CPGC LDAM {tag} seed={seed} done. Best acc {res[\'best_acc\']:.2f}%  ({(time.time()-t_start)/60:.1f} min)")'
    ),
    # 12. CSV
    (
        '    csv_path = os.path.join(OUTPUT_DIR, "CPGC_BS_no_sampler_10种子补跑.csv")',
        '    csv_path = os.path.join(OUTPUT_DIR, "CPGC_LDAM_3种子.csv")'
    ),
    (
        '        w.writerow(["config", "mean", "std", "seed789", "seed2024", "seed7", "seed314", "seed999", "seed55", "seed1337"])',
        '        w.writerow(["config", "mean", "std", "seed42", "seed123", "seed456"])'
    ),
]

for old, new in pairs:
    if old not in s:
        print("[FAIL] 未找到替换片段:")
        print(old[:120].replace("\n", "\\n"))
        raise SystemExit(1)
    s = s.replace(old, new, 1)

with io.open(DST, "w", encoding="utf-8") as f:
    f.write(s)

print(f"OK 生成 {DST}")
