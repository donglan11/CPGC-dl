# 生成 "关 sampler + BS" 10 种子补跑脚本（补 7 个缺失种子）
src = open("CPGC_BS_补跑.py", encoding="utf-8").read()

# 1. RUN_QUEUE 只跑 no_sampler 的 7 个缺失种子（42/123/456 已完成）
old_run = '''    RUN_QUEUE = [
        (False, "no_sampler", [456]),
        (True,  "with_sampler", [42, 123, 456]),
    ]'''
new_run = '''    RUN_QUEUE = [
        (False, "no_sampler", [789, 2024, 7, 314, 999, 55, 1337]),
    ]'''
assert old_run in src, "RUN_QUEUE 未匹配"
src = src.replace(old_run, new_run)

# 2. CSV 文件名
old_csv = 'csv_path = os.path.join(OUTPUT_DIR, "CPGC_BS_补跑结果.csv")'
new_csv = 'csv_path = os.path.join(OUTPUT_DIR, "CPGC_BS_no_sampler_10种子补跑.csv")'
assert old_csv in src, "CSV 文件名未匹配"
src = src.replace(old_csv, new_csv)

# 3. CSV 表头
old_header = 'w.writerow(["config", "mean", "std", "seed42", "seed123", "seed456"])'
new_header = 'w.writerow(["config", "mean", "std", "seed789", "seed2024", "seed7", "seed314", "seed999", "seed55", "seed1337"])'
assert old_header in src, "CSV 表头未匹配"
src = src.replace(old_header, new_header)

open("CPGC_BS_no_sampler_10种子补跑.py", "w", encoding="utf-8").write(src)
print("已生成 CPGC_BS_no_sampler_10种子补跑.py")
