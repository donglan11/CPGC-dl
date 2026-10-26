# -*- coding: utf-8 -*-
"""batch_size 32 vs 64 基准测试：GC10 冻结版三个模型的显存占用 + 前向/反向耗时。
确认 batch=64 不爆显存、且提速，据此决定 GC10 冻结版 batch_size。
用法: python batchsize_bench.py
"""
import os, sys, time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, "D:/EMA/CPGC实验")
_cpgc_mod = __import__("CPGC_10种子训练")
_cpgc_mod.DINO_WEIGHTS_PATH = "D:/EMA/DINO预训练实验/dino_resnet50_pretrain.pth"
load_dino_backbone = _cpgc_mod.load_dino_backbone
supcon_loss = _cpgc_mod.supcon_loss
CPGC = _cpgc_mod.CPGC
from torchvision.models import resnet50

NUM_CLASSES = 10

def freeze_backbone(model, bn_eval=True):
    for p in model.parameters():
        p.requires_grad = False
    if bn_eval:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    return model

def build_dino(nc):
    m = load_dino_backbone()
    m.fc = nn.Linear(2048, nc)
    freeze_backbone(m)
    for p in m.fc.parameters(): p.requires_grad = True
    return m

def build_cpgc(nc):
    m = CPGC(num_classes=nc)
    m.backbone = freeze_backbone(m.backbone)
    for p in m.projection.parameters(): p.requires_grad = True
    for p in m.classifier.parameters(): p.requires_grad = True
    m.fusion_alpha.requires_grad = True
    m.log_proto_temp.requires_grad = True
    return m

def build_resnet(nc):
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, nc)
    return m

def bench(name, build_fn, is_cpgc, batch, device, iters=5):
    model = build_fn(NUM_CLASSES).to(device)
    # 冻结版训练态：CPGC backbone eval，头 train
    model.train()
    if is_cpgc:
        model.backbone.eval()
    else:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scaler = GradScaler()
    crit = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch,), device=device)

    # warmup
    for _ in range(2):
        opt.zero_grad()
        with autocast():
            if is_cpgc:
                proj, mlp, ps, fl = model(x, labels=y, epoch=100)
                loss = crit(fl, y) + 0.1 * supcon_loss(proj, y, 0.07)
                if ps is not None: loss = loss + 0.3 * crit(ps, y)
            else:
                loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        with autocast():
            if is_cpgc:
                proj, mlp, ps, fl = model(x, labels=y, epoch=100)
                loss = crit(fl, y) + 0.1 * supcon_loss(proj, y, 0.07)
                if ps is not None: loss = loss + 0.3 * crit(ps, y)
            else:
                loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{name:<8} batch={batch}: 单步 {dt*1000:.0f}ms, 峰值显存 {mem:.0f}MB")
    del model, opt, scaler, x, y
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("无CUDA，跳过"); sys.exit(0)
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, 总显存 {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print("=" * 60)
    for name, fn, is_cpgc in [("DINO冻结", build_dino, False), ("CPGC冻结", build_cpgc, True)]:
        for b in [32, 64]:
            try:
                bench(name, fn, is_cpgc, b, device)
            except Exception as e:
                print(f"{name} batch={b} 报错: {type(e).__name__}: {str(e)[:100]}")
    print("=" * 60)
    print("基准完成")
