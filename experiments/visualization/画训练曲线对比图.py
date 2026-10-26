"""
训练曲线对比图
=============================================
读取 Baseline 和 CPGC 的训练日志，绘制对比图
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_log(filename):
    epochs, train_loss, train_acc, test_loss, test_acc, best_acc = [], [], [], [], [], []
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            test_loss.append(float(row["test_loss"]))
            test_acc.append(float(row["test_acc"]))
            best_acc.append(float(row["best_acc"]))
    return epochs, train_loss, train_acc, test_loss, test_acc, best_acc

# 读取数据
b_epochs, b_tl, b_ta, b_tel, b_tea, b_ba = read_log("Baseline_training_log_seed42.csv")
c_epochs, c_tl, c_ta, c_tel, c_tea, c_ba = read_log("CPGC_training_log_seed42.csv")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Training Curve Comparison (seed=42)", fontsize=16, fontweight='bold')

# 1. Train Loss
axes[0, 0].plot(b_epochs, b_tl, label='ResNet-50 Baseline', color='#1f77b4', linewidth=1.5)
axes[0, 0].plot(c_epochs, c_tl, label='CPGC (Ours)', color='#d62728', linewidth=1.5)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Train Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Test Accuracy
axes[0, 1].plot(b_epochs, b_tea, label='ResNet-50 Baseline', color='#1f77b4', linewidth=1.5)
axes[0, 1].plot(c_epochs, c_tea, label='CPGC (Ours)', color='#d62728', linewidth=1.5)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Test Accuracy (%)')
axes[0, 1].set_title('Test Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Best Accuracy
axes[1, 0].plot(b_epochs, b_ba, label='ResNet-50 Baseline', color='#1f77b4', linewidth=1.5)
axes[1, 0].plot(c_epochs, c_ba, label='CPGC (Ours)', color='#d62728', linewidth=1.5)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Best Accuracy (%)')
axes[1, 0].set_title('Best Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Train Accuracy
axes[1, 1].plot(b_epochs, b_ta, label='ResNet-50 Baseline', color='#1f77b4', linewidth=1.5)
axes[1, 1].plot(c_epochs, c_ta, label='CPGC (Ours)', color='#d62728', linewidth=1.5)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Train Accuracy (%)')
axes[1, 1].set_title('Training Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "训练曲线_对比图.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"训练曲线对比图已保存: {save_path}")

# 打印最终结果对比
print(f"\n{'='*50}")
print(f"最终结果对比")
print(f"{'='*50}")
print(f"Baseline: Best Acc = {b_ba[-1]:.2f}%")
print(f"CPGC:     Best Acc = {c_ba[-1]:.2f}%")
print(f"提升:     {c_ba[-1] - b_ba[-1]:+.2f}%")
