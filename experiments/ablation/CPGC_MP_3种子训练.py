"""
CPGC + 多原型 (Multi-Prototype) (3 种子)
=============================================
单原型: 每类 1 个原型 (10×128)
多原型: 每类 3 个原型 (10×3×128), 取各原型最大相似度作为该类得分

动机: GC10 缺陷形态多样, 多原型覆盖类内多样性
"""
import os, sys, time, csv, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50

SEEDS = [42, 123, 456]
IMG_SIZE = 224
DATA_DIR = "D:/EMA/GC10_dataset"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 100
BATCH_SIZE = 32
BALANCE_CLASSES = True
LAMBDA_SC = 0.1
LAMBDA_PROTO = 0.3
TEMPERATURE = 0.07
NUM_PROTOS = 3  # 每类原型数

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


# ===================== Loss =====================
def supcon_loss(features, labels, temperature=0.07):
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


# ===================== DINO Backbone =====================
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


# ===================== CPGC + 多原型 =====================
class CPGC_MultiProto(nn.Module):
    def __init__(self, num_classes=10, proj_dim=128, num_protos=3):
        super().__init__()
        self.backbone = load_dino_backbone()
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.num_protos = num_protos

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

        # MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

        # 多原型缓存 (10, 3, 128)
        self.register_buffer("prototypes", torch.randn(num_classes, num_protos, proj_dim) * 0.01)
        self.register_buffer("proto_count", torch.zeros(num_classes, num_protos))

        # 可学习参数
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.log_proto_temp = nn.Parameter(torch.tensor(math.log(0.1)))

    def compute_proto_scores(self, proj_features):
        """proj_features: (B, 128) → (B, num_classes) 每类取最大相似度"""
        proj_norm = F.normalize(proj_features, dim=1)  # (B, 128)
        proto_norm = F.normalize(self.prototypes, dim=2)  # (10, 3, 128)
        sim = torch.einsum("bd,ckd->bck", proj_norm, proto_norm)  # (B, 10, 3)
        sim_max = sim.max(dim=2).values  # (B, 10) 每类取最大
        temp = torch.exp(self.log_proto_temp).clamp(min=0.01)
        return sim_max / temp

    def update_prototypes(self, proj_features, labels):
        """hard assignment: 每个样本更新最近的同类原型"""
        with torch.no_grad():
            proj_norm = F.normalize(proj_features, dim=1).float()
            proto_norm = F.normalize(self.prototypes, dim=2)  # 始终 float32

            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                class_feats = proj_norm[mask]  # (N_c, 128)

                # 计算每个样本到该类所有原型的相似度
                sim = class_feats @ proto_norm[c].T  # (N_c, 3)

                # 每个样本分配到最近的 prototype
                assignments = sim.argmax(dim=1)  # (N_c,)

                for p in range(self.num_protos):
                    pmask = assignments == p
                    if pmask.sum() == 0:
                        continue
                    mean_feat = class_feats[pmask].mean(dim=0)
                    count = self.proto_count[c, p]
                    if count == 0:
                        self.prototypes[c, p] = mean_feat
                    else:
                        self.prototypes[c, p] = 0.9 * self.prototypes[c, p] + 0.1 * mean_feat
                    self.proto_count[c, p] += 1

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


# ===================== 训练 =====================
def train_one(seed, train_loader, test_loader, num_classes, device):
    set_seed(seed)
    model = CPGC_MultiProto(num_classes=num_classes, num_protos=NUM_PROTOS).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
            bsz = labels.size(0)

            optimizer.zero_grad()
            with autocast():
                proj, mlp_logits, proto_scores, final_logits = model(
                    images, labels=labels, epoch=epoch
                )
                loss_ce = criterion_ce(final_logits, labels)
                loss_sc = supcon_loss(proj, labels, TEMPERATURE)
                loss = loss_ce + LAMBDA_SC * loss_sc

                if proto_scores is not None:
                    loss_proto = criterion_ce(proto_scores, labels)
                    loss = loss + LAMBDA_PROTO * loss_proto
                    tr_proto += loss_proto.item() * bsz

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch >= 10:
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
                    _, _, _, final_logits = model(images, epoch=EPOCHS)
                _, predicted = final_logits.max(1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.numpy())
        preds_np = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels)
        te_acc = 100.0 * (preds_np == labels_np).sum() / len(labels_np)
        elapsed = time.time() - t0

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            alpha_val = torch.sigmoid(model.fusion_alpha).item()
            temp_val = torch.exp(model.log_proto_temp).item()
            print(f"  [CPGC-MP seed={seed}] Ep {epoch:>3}/{EPOCHS} "
                  f"CE={tr_ce/tr_total:.3f} SC={tr_sc/tr_total:.3f} "
                  f"α={alpha_val:.3f} τ={temp_val:.3f} "
                  f"Test {te_acc:.2f}%  Best {best_acc:.2f}%  ({elapsed:.1f}s)")

        torch.cuda.empty_cache()

    model_path = os.path.join(OUTPUT_DIR, f"CPGC_MP_seed{seed}_best.pth")
    torch.save(best_state, model_path)
    del model
    torch.cuda.empty_cache()
    return {"best_acc": best_acc}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}  Epochs: {EPOCHS}  Protos/class: {NUM_PROTOS}")

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
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=tf)
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
    num_classes = len(train_set.classes)
    print(f"Classes: {train_set.classes}")

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
        res = train_one(seed, train_loader, test_loader, num_classes, device)
        results[seed] = res
        print(f"<<< CPGC-MP seed={seed} done. Best acc {res['best_acc']:.2f}%  ({(time.time()-t_start)/60:.1f} min)")

    accs = [results[s]["best_acc"] for s in SEEDS]
    print(f"\n{'='*50}")
    print(f"CPGC + 多原型 (每类{NUM_PROTOS}个) (3 种子)")
    print(f"{'='*50}")
    for s in SEEDS:
        print(f"  seed={s}: {results[s]['best_acc']:.2f}%")
    print(f"  mean={np.mean(accs):.2f}%  std={np.std(accs, ddof=1):.2f}%")
    print(f"  vs baseline 89.81%: diff={np.mean(accs)-89.81:+.2f}%")
    print(f"  vs CPGC 91.99%: diff={np.mean(accs)-91.99:+.2f}%")

    csv_path = os.path.join(OUTPUT_DIR, "CPGC_MP_3种子结果.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "best_acc"])
        for s in SEEDS:
            w.writerow([s, round(results[s]["best_acc"], 2)])
        w.writerow([])
        w.writerow(["mean", round(np.mean(accs), 2)])
        w.writerow(["std", round(np.std(accs, ddof=1), 2)])
        w.writerow(["vs_baseline", round(np.mean(accs) - 89.81, 2)])
        w.writerow(["vs_cpgc", round(np.mean(accs) - 91.99, 2)])
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()