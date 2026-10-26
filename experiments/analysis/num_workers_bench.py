# -*- coding: utf-8 -*-
"""num_workers 基准测试：验证 GC10 数据加载提速 + 是否触发 error 1455。
只测 DataLoader 吞吐，不训练。跑完后据此决定 GC10 冻结版 num_workers 设多少。
用法: python num_workers_bench.py
"""
import os, sys, time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = "D:/EMA/GC10_dataset"
IMG_SIZE = 224

def make_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def bench(num_workers, n_batches=20):
    t = make_train_transform()
    ds = datasets.ImageFolder(os.path.join(ROOT, "train"), transform=t)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=num_workers,
                    pin_memory=True, persistent_workers=(num_workers > 0))
    start = time.time()
    cnt = 0
    for i, (x, y) in enumerate(dl):
        cnt += 1
        if cnt >= n_batches:
            break
    dt = time.time() - start
    print(f"num_workers={num_workers}: {n_batches} batches 耗时 {dt:.2f}s "
          f"({n_batches*32/dt:.0f} 图/s)")
    del dl, ds

if __name__ == "__main__":
    print("CPU 逻辑核数:", os.cpu_count())
    for nw in [0, 2, 4]:
        try:
            bench(nw)
        except Exception as e:
            print(f"num_workers={nw} 报错: {type(e).__name__}: {e}")
    print("基准测试完成")
