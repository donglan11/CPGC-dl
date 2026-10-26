# -*- coding: utf-8 -*-
"""生成 CPGC_BS_补跑.py：复制原脚本，num_workers=0，只跑缺失的 4 个种子。"""
src_path = r"D:\EMA\CPGC实验\CPGC_BS_3种子训练.py"
dst_path = r"D:\EMA\CPGC实验\CPGC_BS_补跑.py"

src = open(src_path, encoding="utf-8").read()

# 1) num_workers=2 -> 0（3 处）
assert src.count("num_workers=2") == 3, f"num_workers=2 出现 {src.count('num_workers=2')} 次"
src = src.replace("num_workers=2", "num_workers=0")

# 2) 替换 main 双循环为补跑队列
old = '''    all_rows = []
    for cfg in CONFIGS:
        tag = cfg["tag"]
        use_sampler = cfg["use_sampler"]'''
new = '''    # 补跑队列：no_sampler seed456（上次 error1455 崩溃）+ with_sampler 3 种子（未跑）
    RUN_QUEUE = [
        (False, "no_sampler", [456]),
        (True,  "with_sampler", [42, 123, 456]),
    ]
    all_rows = []
    for use_sampler, tag, seeds in RUN_QUEUE:'''
assert old in src, "[FAIL] 锚点2 未命中"
src = src.replace(old, new, 1)

# 3) 循环体 for seed in SEEDS -> for seed in seeds
old2 = '''        results = {}
        for seed in SEEDS:'''
new2 = '''        results = {}
        for seed in seeds:'''
assert old2 in src, "[FAIL] 锚点3 未命中"
src = src.replace(old2, new2, 1)

# 4) accs 汇总 for s in SEEDS -> for s in seeds
old3 = '''        accs = [results[s]["best_acc"] for s in SEEDS]
        mean = float(np.mean(accs))
        std = float(np.std(accs, ddof=1))
        all_rows.append([tag, round(mean, 2), round(std, 2)] + [round(results[s]["best_acc"], 2) for s in SEEDS])'''
new3 = '''        accs = [results[s]["best_acc"] for s in seeds]
        mean = float(np.mean(accs))
        std = float(np.std(accs, ddof=1))
        all_rows.append([tag, round(mean, 2), round(std, 2)] + [round(results[s]["best_acc"], 2) for s in seeds])'''
assert old3 in src, "[FAIL] 锚点4 未命中"
src = src.replace(old3, new3, 1)

# 5) CSV 文件名
assert '"CPGC_BS_3种子结果.csv"' in src, "[FAIL] CSV 名未命中"
src = src.replace('"CPGC_BS_3种子结果.csv"', '"CPGC_BS_补跑结果.csv"')

# 6) docstring 说明
src = src.replace(
    'CPGC + Balanced Softmax 对照实验 (3 种子 x 2 配置)',
    'CPGC + Balanced Softmax 补跑 (num_workers=0, 修复 error1455)'
)

open(dst_path, "w", encoding="utf-8").write(src)
print("生成完成:", dst_path)
print("RUN_QUEUE 检查:", "RUN_QUEUE" in src)
print("num_workers=0 次数:", src.count("num_workers=0"))
