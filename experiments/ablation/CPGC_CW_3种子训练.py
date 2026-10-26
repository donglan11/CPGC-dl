"""
CPGC 参数调优 — 修改 LAMBDA_PROTO 后运行
=============================================
用法: 修改下方 LAMBDA_PROTO 值，然后运行
"""
import os, sys, time, csv, random, math, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50

# ===== 调参 =====
LAMBDA_PROTO = 0.15   # 当前试的值
LAMBDA_SC = 0.1       # 保持不变
LR = 1e-3             # 保持不变
TAG = f"LP{LAMBDA_PROTO}"  # 输出文件名标签

# ===== 固定配置 =====
SEEDS = [42, 123, 456]
IMG_SIZE = 224
DATA_DIR = "D:/EMA/GC10_dataset"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 100
BATCH_SIZE = 32
BALANCE_CLASSES = True
TEMPERATURE = 0.07
CLASS_WEIGHTED = True  # 类加权 SupCon
TAG = "CW"

DINO_WEIGHTS_PATH = "D:/EMA/DINO预训练实验/dino_resnet50_pretrain.pth"
DINO_WEIGHTS_URLS = [
    "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain.pth",
    "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def supcon_loss(features, labels, temperature=0.07, class_weights=None):
    """
    监督对比损失 (支持类加权)
    class_weights: (N,) 每个样本的权重，None 表示等权重
    """
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
    loss_per_sample = -(pos_mask * log_prob).sum(dim=1).div(pos_count)  # (N,)

    # 类加权
    if class_weights is not None:
        class_weights = class_weights.to(device)
        loss_per_sample = loss_per_sample * class_weights
        return loss_per_sample.sum() / class_weights.sum()
    return loss_per_sample.mean()


def load_dino_backbone():
    if not os.path.exists(DINO_WEIGHTS_PATH):
        print("Downloading DINO weights...")
        for url in DINO_WEIGHTS_URLS:
            try:
                torch.hub.download_url_to_file(url, DINO_WEIGHTS_PATH)
                break
            except Exception:
                continue
    checkpoint = torch.load(DINO_WEIGHTS_PATH, map_location="cpu")
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
    print(f"Loaded DINO backbone ({len(filtered)} params)")
    return model


class CPGC(nn.Module):
    def __init__(self, num_classes=10, proj_dim=128):
        super().__init__()
        self.backbone = load_dino_backbone()
        self.num_classes = num_classes
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
        self.log_proto_temp = nn.Parameter(torch.tensor(math.log(0.1)))

    def compute_proto_scores(self, proj_features):
        proj_norm = F.normalize(proj_features, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim = proj_norm @ proto_norm.T
        temp = torch.exp(self.log_proto_temp).clamp(min=0.01)
        return sim / temp

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
                    self.prototypes[c] = 0.9 * self.prototypes[c] + 0.1 * class_mean
                self.proto_count[c] += 1

    def forward(self, x, labels=None, epoch=None):
        feat = self.backbone(x)
        proj = self.projection(feat)
        mlp_logits = self.classifier(feat)
        if epoch is not None and epoch >= 10:
            proto_scores = self.compute_proto_scores(proj)
            alpha = torch.sigmoid(self.fusion_alpha)
            final_logits = alpha * mlp_logits + (1 - alpha) * proto_scores
        else:
            proto_scores = None
            final_logits = mlp_logits
        return proj, mlp_logits, proto_scores, final_logits


def train_one(seed, train_loader, test_loader, num_classes, device, class_counts=None):
    set_seed(seed)
    model = CPGC(num_classes=num_classes).to(device)

    # 类加权系数
    if CLASS_WEIGHTED and class_counts is not None:
        total = sum(class_counts)
        weights = torch.tensor([total / (num_classes * c) for c in class_counts], dtype=torch.float).to(device)
        print(f"  Class weights: {[f'{w:.2f}' for w in weights.cpu().tolist()]}")
    else:
        weights = None
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    best_acc = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_ce, tr_sc, tr_proto, tr_correct, tr_total = 0.0, 0.0, 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # 计算当前 batch 的类权重
            if CLASS_WEIGHTED and weights is not None:
                class_weights = weights[labels]  # (B,)
            else:
                class_weights = None

            optimizer.zero_grad()
            with autocast():
                proj, mlp_logits, proto_scores, final_logits = model(images, epoch=epoch)
                loss_ce = criterion_ce(final_logits, labels)
                loss_sc = supcon_loss(proj, labels, TEMPERATURE, class_weights)
                loss = loss_ce + LAMBDA_SC * loss_sc
                if proto_scores is not None:
                    loss_proto = criterion_ce(proto_scores, labels)
                    loss = loss + LAMBDA_PROTO * loss_proto
                    tr_proto += loss_proto.item() * labels.size(0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if epoch >= 10:
                model.update_prototypes(proj.detach(), labels)
            tr_ce += loss_ce.item() * labels.size(0)
            tr_sc += loss_sc.item() * labels.size(0)
            _, predicted = final_logits.max(1)
            tr_total += labels.size(0)
            tr_correct += predicted.eq(labels).sum().item()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                with autocast():
                    _, _, _, final_logits = model(images, epoch=EPOCHS)
                _, predicted = final_logits.max(1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.numpy())
        preds_np = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels)
        te_acc = 100.0 * (preds_np == labels_np).sum() / len(labels_np)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{TAG} seed={seed}] Ep {epoch:>3}/{EPOCHS} "
                  f"CE={tr_ce/tr_total:.3f} SC={tr_sc/tr_total:.3f} "
                  f"Test {te_acc:.2f}% Best {best_acc:.2f}% ({time.time()-t0:.1f}s)")
        torch.cuda.empty_cache()

    torch.save(best_state, os.path.join(OUTPUT_DIR, f"CPGC_{TAG}_seed{seed}_best.pth"))
    del model
    torch.cuda.empty_cache()
    return {"best_acc": best_acc}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: LAMBDA_PROTO={LAMBDA_PROTO}  LAMBDA_SC={LAMBDA_SC}  LR={LR}  TAG={TAG}")

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=tf)
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
    num_classes = len(train_set.classes)
    class_counts = [0] * num_classes
    for _, label in train_set.samples:
        class_counts[label] += 1
    if BALANCE_CLASSES:
        sample_weights = [1.0 / class_counts[label] for _, label in train_set.samples]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    results = {}
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        t_start = time.time()
        res = train_one(seed, train_loader, test_loader, num_classes, device, class_counts)
        results[seed] = res
        print(f"<<< {TAG} seed={seed} done. Best acc {res['best_acc']:.2f}% ({(time.time()-t_start)/60:.1f}min)")

    accs = [results[s]["best_acc"] for s in SEEDS]
    print(f"\n{'='*50}")
    print(f"CPGC {TAG} (3 seeds)")
    print(f"{'='*50}")
    for s in SEEDS:
        print(f"  seed={s}: {results[s]['best_acc']:.2f}%")
    print(f"  mean={np.mean(accs):.2f}% std={np.std(accs, ddof=1):.2f}%")
    print(f"  vs CPGC 91.99%: diff={np.mean(accs)-91.99:+.2f}%")

    csv_path = os.path.join(OUTPUT_DIR, f"CPGC_{TAG}_结果.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "best_acc"])
        for s in SEEDS:
            w.writerow([s, round(results[s]["best_acc"], 2)])
        w.writerow([])
        w.writerow(["mean", round(np.mean(accs), 2)])
        w.writerow(["std", round(np.std(accs, ddof=1), 2)])
        w.writerow(["vs_cpgc_9199", round(np.mean(accs) - 91.99, 2)])
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()