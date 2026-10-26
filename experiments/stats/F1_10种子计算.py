"""
计算 Baseline 和 CPGC 的 F1 分数（10种子）
=============================================
加载已训练好的模型，在测试集上计算 Macro-F1
"""
import os, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, precision_score, recall_score

SEEDS = [42, 123, 456, 789, 2024, 7, 314, 999, 55, 1337]
IMG_SIZE = 224
DATA_DIR = "D:/EMA/GC10_dataset"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 32

DINO_WEIGHTS_PATH = "D:/EMA/DINO预训练实验/dino_resnet50_pretrain.pth"

# CPGC 模型权重路径
CPGC_WEIGHTS_DIR = OUTPUT_DIR  # CPGC_10seed_seed{seed}_best.pth


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========== Baseline 模型 ==========
class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 使用 IMAGENET1K_V1 以匹配训练脚本的权重结构
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ========== CPGC 模型 ==========
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
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes),
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

    def forward(self, x, epoch=100):
        feat = self.backbone(x)
        proj = self.projection(feat)
        mlp_logits = self.classifier(feat)
        proto_scores = self.compute_proto_scores(proj)
        alpha = torch.sigmoid(self.fusion_alpha)
        final_logits = alpha * mlp_logits + (1 - alpha) * proto_scores
        return final_logits


def evaluate_baseline(seed, test_loader, num_classes, device):
    set_seed(seed)
    model = ResNet50Baseline(num_classes=num_classes).to(device)
    model_path = os.path.join(os.path.dirname(OUTPUT_DIR), "ResNet50基线实验", f"ResNet50_seed{seed}_best.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        # 训练脚本保存的是普通 resnet50 的 key（conv1.weight, fc.weight）
        # 模型使用 self.model 包装，需要加 "model." 前缀
        prefixed = {"model." + k: v for k, v in state_dict.items()}
        model.load_state_dict(prefixed)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with autocast():
                logits = model(images)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    return acc, f1, prec, rec


def evaluate_cpgc(seed, test_loader, num_classes, device):
    set_seed(seed)
    model = CPGC(num_classes=num_classes).to(device)
    model_path = os.path.join(OUTPUT_DIR, f"CPGC_10seed_seed{seed}_best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with autocast():
                logits = model(images)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    return acc, f1, prec, rec


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
    num_classes = len(test_set.classes)
    print(f"Classes: {test_set.classes}")
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    baseline_results = []
    cpgc_results = []

    print(f"\n{'='*80}")
    print(f"{'Seed':>8} {'B-Acc':>8} {'B-F1':>8} {'B-Prec':>8} {'B-Rec':>8} {'C-Acc':>8} {'C-F1':>8} {'C-Prec':>8} {'C-Rec':>8}")
    print(f"{'='*80}")

    for seed in SEEDS:
        b_acc, b_f1, b_prec, b_rec = evaluate_baseline(seed, test_loader, num_classes, device)
        c_acc, c_f1, c_prec, c_rec = evaluate_cpgc(seed, test_loader, num_classes, device)
        baseline_results.append((seed, b_acc, b_f1, b_prec, b_rec))
        cpgc_results.append((seed, c_acc, c_f1, c_prec, c_rec))
        print(f"{seed:>8} {b_acc:>7.2f}% {b_f1:>7.2f}% {b_prec:>7.2f}% {b_rec:>7.2f}% {c_acc:>7.2f}% {c_f1:>7.2f}% {c_prec:>7.2f}% {c_rec:>7.2f}%")

    b_accs = [r[1] for r in baseline_results]
    b_f1s = [r[2] for r in baseline_results]
    b_precs = [r[3] for r in baseline_results]
    b_recs = [r[4] for r in baseline_results]
    c_accs = [r[1] for r in cpgc_results]
    c_f1s = [r[2] for r in cpgc_results]
    c_precs = [r[3] for r in cpgc_results]
    c_recs = [r[4] for r in cpgc_results]

    print(f"\n{'='*80}")
    print(f"Baseline: Acc={np.mean(b_accs):.2f}±{np.std(b_accs,ddof=1):.2f}  F1={np.mean(b_f1s):.2f}±{np.std(b_f1s,ddof=1):.2f}  Prec={np.mean(b_precs):.2f}±{np.std(b_precs,ddof=1):.2f}  Rec={np.mean(b_recs):.2f}±{np.std(b_recs,ddof=1):.2f}")
    print(f"CPGC:     Acc={np.mean(c_accs):.2f}±{np.std(c_accs,ddof=1):.2f}  F1={np.mean(c_f1s):.2f}±{np.std(c_f1s,ddof=1):.2f}  Prec={np.mean(c_precs):.2f}±{np.std(c_precs,ddof=1):.2f}  Rec={np.mean(c_recs):.2f}±{np.std(c_recs,ddof=1):.2f}")
    print(f"提升:     Acc={np.mean(c_accs)-np.mean(b_accs):+.2f}%  F1={np.mean(c_f1s)-np.mean(b_f1s):+.2f}%  Prec={np.mean(c_precs)-np.mean(b_precs):+.2f}%  Rec={np.mean(c_recs)-np.mean(b_recs):+.2f}%")

    # 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, "F1_Precision_Recall_10种子对比.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "baseline_acc", "baseline_f1", "baseline_precision", "baseline_recall",
                     "cpgc_acc", "cpgc_f1", "cpgc_precision", "cpgc_recall"])
        for i, seed in enumerate(SEEDS):
            w.writerow([seed, round(baseline_results[i][1], 2), round(baseline_results[i][2], 2),
                        round(baseline_results[i][3], 2), round(baseline_results[i][4], 2),
                        round(cpgc_results[i][1], 2), round(cpgc_results[i][2], 2),
                        round(cpgc_results[i][3], 2), round(cpgc_results[i][4], 2)])
        w.writerow([])
        w.writerow(["mean", round(np.mean(b_accs), 2), round(np.mean(b_f1s), 2), round(np.mean(b_precs), 2), round(np.mean(b_recs), 2),
                     round(np.mean(c_accs), 2), round(np.mean(c_f1s), 2), round(np.mean(c_precs), 2), round(np.mean(c_recs), 2)])
        w.writerow(["std", round(np.std(b_accs, ddof=1), 2), round(np.std(b_f1s, ddof=1), 2), round(np.std(b_precs, ddof=1), 2), round(np.std(b_recs, ddof=1), 2),
                     round(np.std(c_accs, ddof=1), 2), round(np.std(c_f1s, ddof=1), 2), round(np.std(c_precs, ddof=1), 2), round(np.std(c_recs, ddof=1), 2)])
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
