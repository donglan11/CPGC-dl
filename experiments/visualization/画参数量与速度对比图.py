"""
可视化：参数量与推理速度对比图 (Baseline vs CPGC)
读取 参数量与推理速度对比.csv 中的数据，生成双栏对比图
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ─── 中文字体 ───
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

# ═══════════════════════════════════════════════
# 数据 (来自 参数量与推理速度对比.csv)
# ═══════════════════════════════════════════════

# 参数量 (百万)
backbone_params = 23.51  # ResNet-50 共用 (≈23.5M)
baseline_head = 0.02     # Linear(2048->10) = 20,490
cpgc_proj = 1.12         # 投影头 2048->512->128
cpgc_cls = 2.63          # MLP分类头 2048->1024->512->10
cpgc_fusion = 0.000002   # alpha + temp = 2 params

baseline_total = 23.53   # 23,528,522
cpgc_total = 27.25       # 27,251,916

# 推理速度
baseline_latency = 25.37  # ms
cpgc_latency = 25.32      # ms
baseline_fps = 1261.1
cpgc_fps = 1263.6

# ═══════════════════════════════════════════════
# 画图
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Baseline vs CPGC: 参数量与推理速度对比', fontsize=16, fontweight='bold', y=0.98)

colors = ['#4472C4', '#ED7D31']  # 蓝=Baseline, 橙=CPGC

# ─── 图1: 总参数量对比 ───
ax1 = axes[0]
bars1 = ax1.bar(['Baseline\n(ResNet-50)', 'CPGC\n(DINO+CPGC)'],
                [baseline_total, cpgc_total],
                color=colors, width=0.5, edgecolor='white', linewidth=1.5)
ax1.set_ylabel('参数量 (M)', fontsize=12)
ax1.set_title('总参数量', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 32)
for bar, val in zip(bars1, [baseline_total, cpgc_total]):
    ax1.text(bar.get_x(), bar.get_height() + 0.5, f'{val:.2f}M',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
# 标注增量
delta = cpgc_total - baseline_total
ax1.annotate(f'+{delta:.2f}M\n(+{delta/baseline_total*100:.1f}%)',
             xy=(1, cpgc_total), xytext=(1.35, cpgc_total - 3),
             fontsize=10, color='#C00000', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ─── 图2: 参数分解 (堆叠柱状图) ───
ax2 = axes[1]
x2 = np.arange(2)
w2 = 0.45

# Baseline: backbone + head
baseline_parts = [backbone_params, baseline_head]
# CPGC: backbone + proj + cls + fusion
cpgc_parts = [backbone_params, cpgc_proj, cpgc_cls, cpgc_fusion + 0.01]  # fusion极小，给最小可见高度

labels_parts = ['Backbone\n(ResNet-50)', '分类头']
bottom_b = 0
bars_b = ax2.bar(x2[0], baseline_parts[0], w2, color='#4472C4', label='Backbone', edgecolor='white')
bars_b2 = ax2.bar(x2[0], baseline_parts[1], w2, bottom=baseline_parts[0],
                  color='#5B9BD5', label='分类头', edgecolor='white')

# CPGC 分解
cpgc_colors = ['#ED7D31', '#F4B183', '#E2A036', '#FFD966']
cpgc_labels = ['Backbone', '投影头', 'MLP分类头', '融合参数']
bottom_c = 0
for i, (val, color, label) in enumerate(zip(cpgc_parts, cpgc_colors, cpgc_labels)):
    ax2.bar(x2[1], val, w2, bottom=bottom_c, color=color, label=label, edgecolor='white')
    if val > 0.1:  # 只标注够大的块
        ax2.text(x2[1], bottom_c + val/2, f'{val:.2f}M',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    bottom_c += val

ax2.set_xticks(x2)
ax2.set_xticklabels(['Baseline', 'CPGC'], fontsize=11)
ax2.set_ylabel('参数量 (M)', fontsize=12)
ax2.set_title('参数构成分解', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 32)
ax2.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ─── 图3: 推理延迟 + FPS 双轴 ───
ax3 = axes[2]
x3 = np.arange(2)
w3 = 0.35

# 左轴: 延迟(ms)
bars_lat = ax3.bar(x3 - w3/2, [baseline_latency, cpgc_latency], w3,
                   color=colors, edgecolor='white', linewidth=1.5)
ax3.set_ylabel('推理延迟 (ms)', fontsize=12, color='#333333')
ax3.set_ylim(20, 30)
ax3.tick_params(axis='y', labelcolor='#333333')
for bar, val in zip(bars_lat, [baseline_latency, cpgc_latency]):
    ax3.text(bar.get_x(), bar.get_height() + 0.15, f'{val:.2f}ms',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 右轴: FPS
ax3r = ax3.twinx()
bars_fps = ax3r.bar(x3 + w3/2, [baseline_fps, cpgc_fps], w3,
                    color=['#A9D18E', '#A9D18E'], edgecolor='white', linewidth=1.5,
                    hatch='//', alpha=0.8)
ax3r.set_ylabel('吞吐量 (FPS)', fontsize=12, color='#548235')
ax3r.set_ylim(1000, 1500)
ax3r.tick_params(axis='y', labelcolor='#548235')
for bar, val in zip(bars_fps, [baseline_fps, cpgc_fps]):
    ax3r.text(bar.get_x(), bar.get_height() + 15, f'{val:.1f}',
              ha='center', va='bottom', fontsize=11, fontweight='bold', color='#548235')

ax3.set_xticks(x3)
ax3.set_xticklabels(['Baseline', 'CPGC'], fontsize=11)
ax3.set_title('推理速度 (batch=32)', fontsize=13, fontweight='bold')
ax3.spines['top'].set_visible(False)

# 标注"几乎零开销"
ax3.annotate('开销仅 -0.2%\n(统计噪声)',
             xy=(1, cpgc_latency), xytext=(0.5, 21.5),
             fontsize=10, color='#548235', fontweight='bold',
             ha='center',
             arrowprops=dict(arrowstyle='->', color='#548235', lw=1.5))

plt.tight_layout(rect=[0, 0, 1, 0.94])

# ─── 保存 ───
out_path = os.path.join(OUTPUT_DIR, '参数量与推理速度_对比图.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'已保存: {out_path}')
