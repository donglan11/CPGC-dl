"""
CPGC 极端 α 分析
=================
目标：验证融合（正常 α）是否优于任一单分支（α=0 纯原型 / α=1 纯 MLP）
数据：10 个已保存的 CPGC 权重，只推理不训练

自校验命门：
  1. 推理路径与 F1_10种子计算.py 的 evaluate_cpgc 逐字一致
  2. "正常 α"的 10 个种子 acc 必须与 CPGC_10种子结果.csv 完全对齐
     有任何种子对不上就停，查加载/推理链
  3. 报告逐种子结果 + 配对 t 检验（α=0 vs 融合, α=1 vs 融合）
"""
import os, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score
from scipy import stats

SEEDS = [42, 123, 456, 789, 2024, 7, 314, 999, 55, 1337]
IMG_SIZE = 224
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 32

# ===== 路径兼容：D:/EMA 或 D:/EMA =====
_PROJECT_ROOTS = ["D:/EMA", "D:/EMA"]
_PROJECT_ROOT = None
for root in _PROJECT_ROOTS:
    if os.path.exists(os.path.join(root, "GC10_dataset/test")):
        _PROJECT_ROOT = root
        break
if _PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"GC10_dataset/test not found under any of {_PROJECT_ROOTS}")

DATA_DIR = os.path.join(_PROJECT_ROOT, "GC10_dataset")
DINO_WEIGHTS_PATH = os.path.join(
    _PROJECT_ROOT, "DINO预训练实验", "dino_resnet50_pretrain.pth")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========== CPGC 模型（与 F1_10种子计算.py 逐字一致，只加 alpha_override） ==========
class CPGC(nn.Module):
    def __init__(self, num_classes=10, proj_dim=128):
        super().__init__()
        checkpoint = torch.load(DINO_WEIGHTS_PATH, map_location="cpu")
        if isinstance(checkpoint, dict) and 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        else:
            state_dict = checkpoint
        model = resnet50()
        model.fc = nn.Identity()
        filtered = {k[7:] if k.startswith('module.') else k: v
                    for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(filtered, strict=False)

        self.backbone = model
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        self.projection = nn.Sequential(
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self.register_buffer("prototypes", torch.randn(num_classes, proj_dim) * 0.01)
        self.register_buffer("proto_count", torch.zeros(num_classes))
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.log_proto_temp = nn.Parameter(torch.tensor(0.1).log())

    def compute_proto_scores(self, proj_features):
        proj_norm = F.normalize(proj_features, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim = proj_norm @ proto_norm.T
        temp = torch.exp(self.log_proto_temp).clamp(min=0.01)
        return sim / temp

    def forward(self, x, alpha_override=None):
        """
        alpha_override:
          None  → 使用 sigmoid(fusion_alpha)，正常融合
          0.0   → 纯原型（proto_scores 独占）
          1.0   → 纯 MLP（mlp_logits 独占）
        """
        feat = self.backbone(x)
        proj = self.projection(feat)
        mlp_logits = self.classifier(feat)
        proto_scores = self.compute_proto_scores(proj)

        if alpha_override is not None:
            alpha = alpha_override
        else:
            alpha = torch.sigmoid(self.fusion_alpha)

        final_logits = alpha * mlp_logits + (1 - alpha) * proto_scores
        return final_logits


# ========== 推理函数（与 F1_10种子计算.py 的 evaluate_cpgc 逐字一致） ==========
def evaluate(seed, test_loader, num_classes, device, alpha_override=None):
    set_seed(seed)
    model = CPGC(num_classes=num_classes).to(device)
    model_path = os.path.join(OUTPUT_DIR, f"CPGC_10seed_seed{seed}_best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device),
                              strict=False)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with autocast():
                logits = model(images, alpha_override=alpha_override)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    return round(acc, 2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_set = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"), transform=test_tf)
    num_classes = len(test_set.classes)
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True)

    # 读已有 CPGC 10 种子结果用于自校验
    expected = {}
    with open(os.path.join(OUTPUT_DIR, "CPGC_10种子结果.csv"),
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["seed"].isdigit():
                expected[int(row["seed"])] = float(row["best_acc"])

    # ===== 跑三种模式 =====
    results = {"alpha0": {}, "alpha1": {}, "normal": {}}

    print("\n" + "=" * 75)
    print(f"{'Seed':>6}  {'alpha=0':>8}  {'alpha=1':>8}  {'normal':>8}"
          f"  {'expected':>8}  {'norm==exp?':>10}")
    print("=" * 75)

    for seed in SEEDS:
        acc0 = evaluate(seed, test_loader, num_classes, device,
                        alpha_override=0.0)
        acc1 = evaluate(seed, test_loader, num_classes, device,
                        alpha_override=1.0)
        accN = evaluate(seed, test_loader, num_classes, device,
                        alpha_override=None)
        results["alpha0"][seed] = acc0
        results["alpha1"][seed] = acc1
        results["normal"][seed] = accN
        exp = expected.get(seed)
        match = "OK" if exp is not None and abs(accN - exp) < 0.01 else "MISMATCH"
        print(f"{seed:>6}  {acc0:>8.2f}  {acc1:>8.2f}  {accN:>8.2f}"
              f"  {exp:>8.2f}  {match:>10}")

    # ===== 自校验 =====
    all_match = True
    for seed in SEEDS:
        exp = expected.get(seed)
        if exp is None or abs(results["normal"][seed] - exp) >= 0.01:
            all_match = False
            print(f"\n[WARN] seed {seed}: normal={results['normal'][seed]:.2f}"
                  f" != expected={exp:.2f}")
    if not all_match:
        print("\n[FATAL] 自校验失败：normal alpha 与已知结果不一致！")
        print("        检查模型加载、推理链后再看 alpha=0/alpha=1 的结果。")
        # 仍然继续输出，但标记为不可信
    else:
        print("\n[PASS] 自校验通过：所有种子 normal alpha 与 CPGC_10种子结果.csv 一致")

    # ===== 汇总统计 =====
    def arr(key):
        return np.array([results[key][s] for s in SEEDS])

    n_acc = arr("normal")
    a0_acc = arr("alpha0")
    a1_acc = arr("alpha1")

    print(f"\n{'=' * 50}")
    print("汇总")
    print(f"{'=' * 50}")
    print(f"{'Mode':>12}  {'Mean Acc':>10}  {'Std':>8}")
    print("-" * 34)
    print(f"{'Normal':>12}  {np.mean(n_acc):>10.2f}  {np.std(n_acc, ddof=1):>8.2f}")
    print(f"{'alpha=1 (MLP)':>12}  {np.mean(a1_acc):>10.2f}  {np.std(a1_acc, ddof=1):>8.2f}")
    print(f"{'alpha=0 (Proto)':>12}  {np.mean(a0_acc):>10.2f}  {np.std(a0_acc, ddof=1):>8.2f}")

    # ===== 配对 t 检验 =====
    print(f"\n{'=' * 50}")
    print("配对 t 检验（10 种子，vs Normal 融合）")
    print(f"{'=' * 50}")

    for label, arr_a in [("alpha=1 (纯 MLP)", a1_acc),
                          ("alpha=0 (纯原型)", a0_acc)]:
        diff = arr_a - n_acc
        t, p = stats.ttest_rel(arr_a, n_acc)
        ci = stats.t.interval(0.95, df=len(diff) - 1,
                              loc=np.mean(diff), scale=stats.sem(diff))
        print(f"\n{label} vs Normal:")
        print(f"  均值差值 = {np.mean(diff):+.2f}")
        print(f"  配对 t = {t:.3f}, p = {p:.4f}")
        print(f"  95% CI = [{ci[0]:.2f}, {ci[1]:.2f}]")
        if p < 0.05:
            print(f"  p < 0.05 → {'MLP 分支' if 'MLP' in label else '原型分支'}"
                  f" {'显著低于' if np.mean(diff) < 0 else '显著高于'}融合")
        else:
            print(f"  p >= 0.05 → 差异未达统计显著")

    # ===== 保存 CSV =====
    csv_path = os.path.join(OUTPUT_DIR, "CPGC_极端alpha分析.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "alpha0_proto", "alpha1_mlp", "normal_fusion"])
        for s in SEEDS:
            w.writerow([s, results["alpha0"][s], results["alpha1"][s],
                        results["normal"][s]])
        w.writerow([])
        w.writerow(["mode", "mean", "std"])
        for label, vals in [("normal", n_acc), ("alpha1_mlp", a1_acc),
                             ("alpha0_proto", a0_acc)]:
            w.writerow([label, round(np.mean(vals), 2),
                        round(np.std(vals, ddof=1), 2)])
        w.writerow([])
        w.writerow(["paired_ttest", "vs_normal"])
        for label, arr_a in [("alpha1_mlp", a1_acc), ("alpha0_proto", a0_acc)]:
            diff = arr_a - n_acc
            t, p = stats.ttest_rel(arr_a, n_acc)
            ci = stats.t.interval(0.95, df=len(diff) - 1,
                                  loc=np.mean(diff), scale=stats.sem(diff))
            w.writerow([label, f"t={t:.3f}, p={p:.4f}, "
                              f"diff={np.mean(diff):+.2f}, "
                              f"95CI=[{ci[0]:.2f},{ci[1]:.2f}]"])
        w.writerow([])
        w.writerow(["self_check",
                     "PASS" if all_match else "FAIL"])
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()