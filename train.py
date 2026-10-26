"""
CPGC —— 原型引导对比学习（Prototype-Guided Contrastive Learning）
长尾钢材表面缺陷分类 · 训练脚本
=====================================================================
复现论文主表（10 种子配对检验）：

    python train.py --data-dir data/GC10-DET --epochs 30 --seeds 10

数据集目录结构需为 ImageFolder 格式：
    data/GC10-DET/train/<class_name>/*.jpg
    data/GC10-DET/test/<class_name>/*.jpg

GC10-DET（10 类钢材表面缺陷）为公开数据集：
    Lv et al., "Deep Metallic Surface Defect Detection: The New Benchmark
    and Detection Network", Sensors, 2020.  DOI: 10.3390/s20061562
    https://github.com/lvxiaoming/GC10-DET

骨干权重（DINO ResNet-50）首次运行时自动下载，无需手动放置。
=====================================================================
"""
import os
import csv
import math
import random
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50
from scipy import stats

# ---------------------------- 默认超参数 ----------------------------
# 与论文正文一致；命令行参数可覆盖
SEEDS = [42, 123, 456, 789, 2024, 7, 314, 999, 55, 1337]
IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 32
BALANCE_CLASSES = True
LAMBDA_SC = 0.1        # 有监督对比损失权重 λ_sup
LAMBDA_PROTO = 0.3     # 原型分类损失权重 λ_proto
TEMPERATURE = 0.07     # SupCon 温度
PROTO_TEMP = 0.1       # 原型检索温度（可学习，此值为初值）
PROTO_EMA = 0.9        # 原型 EMA 动量 m
PROTO_START = 10       # 原型路径自第几个 epoch 启用

DINO_WEIGHTS_URLS = [
    "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain.pth",
    "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="CPGC 训练脚本（原型引导对比学习 · 长尾钢材表面缺陷分类）")
    p.add_argument("--data-dir", default="./data/GC10-DET",
                   help="数据集根目录，需含 train/ 与 test/ 两个子目录")
    p.add_argument("--output-dir", default="./outputs",
                   help="模型权重与结果 CSV 的输出目录")
    p.add_argument("--backbone-weights", default=None,
                   help="DINO ResNet-50 权重路径；默认自动下载到系统缓存")
    p.add_argument("--baseline-csv", default=None,
                   help="基线（ResNet-50）10 种子结果 CSV，用于配对 t 检验；"
                        "格式见 results/README.md")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader 进程数；Windows 下若报错请保持 0")
    p.add_argument("--seeds", type=str, default="",
                   help="以逗号分隔的随机种子列表；留空则用默认 10 种子")
    p.add_argument("--no-balance", action="store_true",
                   help="关闭类别均衡采样器")
    return p.parse_args()


# ---------------------------- 可复现性 ----------------------------
def set_seed(seed):
    """固定全部随机源，保证跨平台可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------- 损失函数 ----------------------------
def supcon_loss(features, labels, temperature=TEMPERATURE):
    """有监督对比损失（SupCon, Khosla et al. 2020）"""
    N = labels.shape[0]
    device = features.device
    features = F.normalize(features, dim=1)
    sim = features @ features.T / temperature
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    pos_mask.fill_diagonal_(0)
    neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float().to(device)
    exp_sim = torch.exp(sim)
    denom = (exp_sim * (pos_mask + neg_mask)).sum(dim=1, keepdim=True)
    log_prob = sim - torch.log(denom + 1e-8)
    pos_count = torch.clamp(pos_mask.sum(dim=1), min=1.0)
    return -(pos_mask * log_prob).sum(dim=1).div(pos_count).mean()


# ---------------------------- 骨干网络 ----------------------------
def load_dino_backbone(weights_path=None):
    """加载 DINO 自监督预训练的 ResNet-50，去掉分类头。

    权重来源：https://github.com/facebookresearch/dino
    """
    if weights_path is None:
        # 沿用 torch.hub 缓存目录
        cache = os.path.join(torch.hub.get_dir(), "checkpoints")
        os.makedirs(cache, exist_ok=True)
        weights_path = os.path.join(cache, "dino_resnet50_pretrain.pth")

    if not os.path.exists(weights_path):
        print(f"[DINO] 权重不存在，开始下载 -> {weights_path}")
        ok = False
        for url in DINO_WEIGHTS_URLS:
            try:
                torch.hub.download_url_to_file(url, weights_path)
                ok = True
                break
            except Exception as e:
                print(f"[DINO] 下载失败（{url}）：{e}")
        if not ok:
            raise RuntimeError("DINO 权重下载失败，请手动下载后通过 "
                               "--backbone-weights 指定路径。")

    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model = resnet50()
    model.fc = nn.Identity()
    filtered = {k[7:] if k.startswith('module.') else k: v
                for k, v in state_dict.items()
                if not k.startswith('fc.')}
    model.load_state_dict(filtered, strict=False)
    print(f"[DINO] 已加载骨干（{len(filtered)} 个张量）")
    return model


# ---------------------------- CPGC 模型 ----------------------------
class CPGC(nn.Module):
    """原型引导对比学习模型。

    结构：
        骨干 (DINO ResNet-50)
          ├─ 投影头 2048→512→128        （供 SupCon 与原型检索）
          ├─ 分类头 2048→1024→512→10     （判别式路径 s_mlp）
          └─ 类别原型库 C×128            （原型路径 s_proto，EMA 更新）

    融合：s_final = α·s_mlp + (1−α)·s_proto，α = σ(θ_α) 可学习
    """

    def __init__(self, num_classes=10, proj_dim=128):
        super().__init__()
        self.backbone = load_dino_backbone(globals().get("BACKBONE_WEIGHTS"))
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

        # 原型为 buffer，不参与梯度；经 EMA 在线更新
        self.register_buffer("prototypes",
                             torch.randn(num_classes, proj_dim) * 0.01)
        self.register_buffer("proto_count", torch.zeros(num_classes))

        # 可学习融合权重与原型温度
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.log_proto_temp = nn.Parameter(torch.tensor(math.log(PROTO_TEMP)))

    # ---- 原型检索得分：余弦相似度 / 温度 ----
    def compute_proto_scores(self, proj_features):
        proj_norm = F.normalize(proj_features, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim = proj_norm @ proto_norm.T
        temp = torch.exp(self.log_proto_temp).clamp(min=0.01)
        return sim / temp

    # ---- 原型 EMA 更新（detach，无梯度回流）----
    def update_prototypes(self, proj_features, labels):
        with torch.no_grad():
            proj_norm = F.normalize(proj_features, dim=1)
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                class_mean = proj_norm[mask].mean(dim=0)
                if self.proto_count[c] == 0:
                    self.prototypes[c] = class_mean
                else:
                    self.prototypes[c] = (
                        PROTO_EMA * self.prototypes[c] +
                        (1 - PROTO_EMA) * class_mean
                    )
                self.proto_count[c] += 1

    def forward(self, x, labels=None, epoch=None):
        feat = self.backbone(x)
        proj = self.projection(feat)
        mlp_logits = self.classifier(feat)

        if epoch is not None and epoch >= PROTO_START:
            proto_scores = self.compute_proto_scores(proj)
            alpha = torch.sigmoid(self.fusion_alpha)
            final_logits = alpha * mlp_logits + (1 - alpha) * proto_scores
        else:
            proto_scores = None
            final_logits = mlp_logits

        return proj, mlp_logits, proto_scores, final_logits


# ---------------------------- 单种子训练 ----------------------------
def train_one(seed, train_loader, test_loader, num_classes, device, args):
    set_seed(seed)
    model = CPGC(num_classes=num_classes).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    best_acc = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_ce, tr_sc, tr_correct, tr_total = 0.0, 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            bsz = labels.size(0)

            optimizer.zero_grad()
            with autocast():
                proj, mlp_logits, proto_scores, final_logits = model(
                    images, labels=labels, epoch=epoch)

                loss_ce = criterion_ce(final_logits, labels)
                loss_sc = supcon_loss(proj, labels)
                loss = loss_ce + LAMBDA_SC * loss_sc

                if proto_scores is not None:
                    loss_proto = criterion_ce(proto_scores, labels)
                    loss = loss + LAMBDA_PROTO * loss_proto

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch >= PROTO_START:
                model.update_prototypes(proj.detach(), labels)

            tr_ce += loss_ce.item() * bsz
            tr_sc += loss_sc.item() * bsz
            _, predicted = final_logits.max(1)
            tr_total += bsz
            tr_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                with autocast():
                    _, _, _, final_logits = model(images, epoch=args.epochs)
                _, predicted = final_logits.max(1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.numpy())
        preds_np = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels)
        te_acc = 100.0 * (preds_np == labels_np).sum() / len(labels_np)

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            alpha_val = torch.sigmoid(model.fusion_alpha).item()
            print(f"  [seed={seed}] Ep {epoch:>3}/{args.epochs} "
                  f"CE={tr_ce/tr_total:.3f} SC={tr_sc/tr_total:.3f} "
                  f"α={alpha_val:.3f} Test {te_acc:.2f}%  "
                  f"Best {best_acc:.2f}%  ({time.time()-t0:.1f}s)")

        torch.cuda.empty_cache()

    model_path = os.path.join(args.output_dir, f"cpgc_seed{seed}_best.pth")
    torch.save(best_state, model_path)
    del model
    torch.cuda.empty_cache()
    return {"best_acc": best_acc, "model": model_path}


# ---------------------------- 主流程 ----------------------------
def main():
    args = parse_args()

    global BACKBONE_WEIGHTS, PROTO_START
    BACKBONE_WEIGHTS = args.backbone_weights

    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(SEEDS)
    args.seeds_list = seeds

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Seeds  : {seeds}   Epochs: {args.epochs}")
    print(f"λ_SC={LAMBDA_SC}  λ_PROTO={LAMBDA_PROTO}  τ={TEMPERATURE}  "
          f"proto_EMA={PROTO_EMA}  proto_start={PROTO_START}")

    # ---- 数据 ----
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_set = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=tf)
    test_set = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=test_tf)
    num_classes = len(train_set.classes)
    print(f"Classes({num_classes}): {train_set.classes}")

    class_counts = [0] * num_classes
    for _, label in train_set.samples:
        class_counts[label] += 1

    if BALANCE_CLASSES and not args.no_balance:
        sample_weights = [1.0 / class_counts[label] for _, label in train_set.samples]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers,
                                  pin_memory=True)
        print("类别均衡采样器：已启用")
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
        print("类别均衡采样器：已关闭")

    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # ---- 逐种子训练 ----
    results = {}
    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        t_start = time.time()
        res = train_one(seed, train_loader, test_loader, num_classes, device, args)
        results[seed] = res
        print(f"<<< seed={seed} 完成，最优 acc {res['best_acc']:.2f}%  "
              f"({(time.time()-t_start)/60:.1f} min)")

    accs = [results[s]["best_acc"] for s in seeds]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs, ddof=1)
    se_acc = std_acc / np.sqrt(len(accs))

    # ---- 与基线做配对 t 检验（可选）----
    baseline_accs = {}
    if args.baseline_csv and os.path.exists(args.baseline_csv):
        with open(args.baseline_csv, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[0] != "mean":
                    try:
                        baseline_accs[int(parts[0])] = float(parts[1])
                    except ValueError:
                        pass
        print(f"[基线] 已读取 {len(baseline_accs)} 个种子：{args.baseline_csv}")

    print(f"\n{'='*60}")
    print(f"CPGC {len(seeds)} 种子结果")
    print(f"{'='*60}")
    for s in seeds:
        b = f"  (baseline: {baseline_accs[s]:.2f}%)" if s in baseline_accs else ""
        print(f"  seed={s:>6}: {results[s]['best_acc']:.2f}%{b}")

    t_stat = p_value = float("nan")
    if baseline_accs and all(s in baseline_accs for s in seeds):
        base_accs = [baseline_accs[s] for s in seeds]
        t_stat, p_value = stats.ttest_rel(accs, base_accs)
        print(f"\n  CPGC mean     = {mean_acc:.2f}%")
        print(f"  Baseline mean = {np.mean(base_accs):.2f}%")
        print(f"  diff          = {mean_acc-np.mean(base_accs):+.2f}pp")
        print(f"  配对 t-test   : t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  {'*** p < 0.05，提升具有统计显著性 ***' if p_value < 0.05 else 'p >= 0.05，提升不具有统计显著性'}")
    else:
        print(f"\n  mean = {mean_acc:.2f}%   std = {std_acc:.2f}%")
        print(f"  SE   = {se_acc:.2f}%")
        print(f"  95% CI = [{mean_acc-1.96*se_acc:.2f}%, {mean_acc+1.96*se_acc:.2f}%]")

    # ---- 落盘 ----
    csv_path = os.path.join(args.output_dir, "cpgc_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "best_acc", "baseline_acc"])
        for s in seeds:
            w.writerow([s, round(results[s]["best_acc"], 2),
                        round(baseline_accs.get(s, float("nan")), 2)])
        w.writerow([])
        w.writerow(["mean", round(mean_acc, 2)])
        w.writerow(["std", round(std_acc, 2)])
        w.writerow(["SE", round(se_acc, 2)])
        w.writerow(["95%CI_lower", round(mean_acc - 1.96*se_acc, 2)])
        w.writerow(["95%CI_upper", round(mean_acc + 1.96*se_acc, 2)])
        if baseline_accs and all(s in baseline_accs for s in seeds):
            w.writerow(["diff", round(mean_acc - np.mean(base_accs), 2)])
            w.writerow(["paired_t", round(t_stat, 4)])
            w.writerow(["p_value", round(p_value, 4)])
    print(f"\n结果已保存：{csv_path}")
    print(f"权重已保存：{args.output_dir}/cpgc_seed*_best.pth")


if __name__ == "__main__":
    main()
