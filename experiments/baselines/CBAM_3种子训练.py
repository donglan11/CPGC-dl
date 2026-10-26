"""
SE / CBAM 注意力对比实验 (3种子)
=============================================
在 ResNet-50 每个 Bottleneck 的 conv3 之后插入 SE 或 CBAM
配置: ImageNet 预训练, 100 epochs, GC10-DET
"""
import os, sys, time, csv, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

ATTN_TYPE = "CBAM"  # 运行前改成 "SE" 或 "CBAM"
SEEDS = [42, 123, 456]
EPOCHS = 100; BATCH_SIZE = 32; IMG_SIZE = 224
DATA_DIR = "D:/EMA/GC10_dataset"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BALANCE_CLASSES = True


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ===================== SE =====================
class SE(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//r, 1), nn.ReLU(),
            nn.Conv2d(channels//r, channels, 1), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)


# ===================== CBAM =====================
class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels//r, 1), nn.ReLU(),
            nn.Conv2d(channels//r, channels, 1), nn.Sigmoid())
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())
    def forward(self, x):
        x = x * self.ca(x)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return x * self.sa(torch.cat([avg, mx], dim=1))


def add_attention_to_resnet(model, attn_type):
    """在每个 Bottleneck 的 conv3 之后插入注意力模块"""
    for name, module in model.named_modules():
        # 找到所有 Bottleneck
        if isinstance(module, nn.Conv2d):
            continue
        # 通过检查是否有 expansion 属性来识别 Bottleneck
        if hasattr(module, 'expansion') and hasattr(module, 'conv3'):
            channels = module.conv3.out_channels
            if attn_type == "SE":
                module.add_module('attention', SE(channels))
            else:
                module.add_module('attention', CBAM(channels))
            # 修改 forward
            orig_forward = module.forward
            def make_new_forward(orig_fwd, attn_mod):
                def new_forward(self, x):
                    identity = x
                    out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
                    out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
                    out = self.conv3(out); out = self.bn3(out)
                    out = attn_mod(out)  # 注意力
                    if self.downsample is not None: identity = self.downsample(x)
                    out += identity; out = self.relu(out)
                    return out
                return new_forward
            module.forward = make_new_forward(orig_forward, module.attention).__get__(module, type(module))
    return model


def build_model(attn_type, num_classes=10):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, num_classes)
    model = add_attention_to_resnet(model, attn_type)
    print(f"[{attn_type}] Model built, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def train_one(seed, model, train_loader, test_loader, device):
    set_seed(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()
    best_acc = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_correct, tr_total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            _, predicted = model(images).max(1)
            tr_total += labels.size(0); tr_correct += predicted.eq(labels).sum().item()
        scheduler.step()

        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                with autocast():
                    _, predicted = model(images).max(1)
                all_p.append(predicted.cpu().numpy()); all_l.append(labels.numpy())
        preds = np.concatenate(all_p); labels = np.concatenate(all_l)
        te_acc = 100.0 * (preds == labels).sum() / len(labels)

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{ATTN_TYPE} seed={seed}] Ep {epoch:>3}/{EPOCHS} Test {te_acc:.2f}% Best {best_acc:.2f}% ({time.time()-t0:.1f}s)")
        torch.cuda.empty_cache()

    torch.save(best_state, os.path.join(OUTPUT_DIR, f"{ATTN_TYPE}_seed{seed}_best.pth"))
    del model; torch.cuda.empty_cache()
    return {"best_acc": best_acc}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Attention: {ATTN_TYPE}")

    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=tf)
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
    num_classes = len(train_set.classes)
    class_counts = [0]*num_classes
    for _, l in train_set.samples: class_counts[l] += 1
    if BALANCE_CLASSES:
        sw = [1.0/class_counts[l] for _, l in train_set.samples]
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    results = {}
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        model = build_model(ATTN_TYPE, num_classes)
        t_start = time.time()
        res = train_one(seed, model, train_loader, test_loader, device)
        results[seed] = res
        print(f"<<< {ATTN_TYPE} seed={seed} done. {res['best_acc']:.2f}% ({(time.time()-t_start)/60:.1f}min)")

    accs = [results[s]["best_acc"] for s in SEEDS]
    print(f"\n{'='*50}\n{ATTN_TYPE} 3种子\n{'='*50}")
    for s in SEEDS: print(f"  seed={s}: {results[s]['best_acc']:.2f}%")
    print(f"  mean={np.mean(accs):.2f}%  std={np.std(accs, ddof=1):.2f}%  vs 89.81%: {np.mean(accs)-89.81:+.2f}%")

    csv_path = os.path.join(OUTPUT_DIR, f"{ATTN_TYPE}_3种子结果.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["seed","best_acc"])
        for s in SEEDS: w.writerow([s, round(results[s]["best_acc"],2)])
        w.writerow([]); w.writerow(["mean", round(np.mean(accs),2)])
        w.writerow(["std", round(np.std(accs,ddof=1),2)])
        w.writerow(["vs_baseline", round(np.mean(accs)-89.81,2)])
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()