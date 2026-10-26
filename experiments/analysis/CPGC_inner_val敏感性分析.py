"""
CPGC 评估协议敏感性分析（inner-val）
==================================================
目的：量化「best-epoch-on-test」的选模乐观幅度，把它变成有界量。

核心设计（四条护栏）：
  护栏1【within-run 相减】单次 run 内逐 epoch 记录 (test_acc, val_macroF1, val_loss)，
        训练结束后回放三种选模规则，均报同一 run 的测试精度：
          A_test = argmax(test_acc) 的 epoch -> 测试精度（偷看测试集选模）
          A_val  = argmax(val_macroF1) 的 epoch -> 测试精度（诚实选模）
          A_last = 末轮 epoch -> 测试精度（不选模）
        乐观幅度 Δ = A_test - A_val（同 run 相减，训练数据缩减量相互抵消）。
  护栏2【选模信号】用 val 宏平均 F1 选 epoch（连续信号+类别均衡），不用 val 总体准确率。
  护栏3【预算隔离】3 种子（42/123/456），与 LOO / 冻结表同口径；任何数字不得与 10 种子 91.76 并列。
  护栏4【判读标准前置】JUDGE_THRESHOLDS 在跑之前写死，先定标准再看结果。

同源性保证：直接 import 主实验脚本 CPGC_10种子训练.py 的 CPGC 类与 supcon_loss，
            确保网络结构、原型机制、损失组装与主实验完全一致（不重写、不抄错）。
数据划分：从 train 分层抽 15% 作 val，剩余 85% 作 train'。test 保持 466 张不动。
"""
import os, sys, time, csv, random, math, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- 从主实验脚本 import 网络与损失（保证同源） ----------------
_spec = importlib.util.spec_from_file_location(
    "main_train", os.path.join(HERE, "CPGC_10种子训练.py"))
_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main)   # 只执行定义，不触发其 main()

CPGC = _main.CPGC
supcon_loss = _main.supcon_loss
set_seed = _main.set_seed
print("已从 CPGC_10种子训练.py 导入 CPGC / supcon_loss / set_seed（保证同源）")

# ---------------- 配置 ----------------
SEEDS = [42, 123, 456]              # 护栏3：3 种子，与 LOO/冻结表同口径
VAL_RATIO = 0.15                    # 15% 分层抽验证集
IMG_SIZE = 224
DATA_DIR = "D:/EMA/GC10_dataset"
OUTPUT_DIR = HERE
EPOCHS = 100
BATCH_SIZE = 32
BALANCE_CLASSES = True
LAMBDA_SC = 0.1
LAMBDA_PROTO = 0.3
TEMPERATURE = 0.07
PROTO_START = 10

# 护栏4：判读标准前置
JUDGE_THRESHOLDS = {"negligible": 0.5, "moderate": 1.5}

CSV_PATH = os.path.join(OUTPUT_DIR, "inner_val_sensitivity.csv")
TRACE_PATH = os.path.join(OUTPUT_DIR, "inner_val_trace.csv")


def stratified_split(dataset, val_ratio, seed):
    """按类别分层抽 val。返回 train_idx, val_idx。"""
    rng = random.Random(seed)
    targets = [s[1] for s in dataset.samples]
    by_class = {}
    for i, t in enumerate(targets):
        by_class.setdefault(t, []).append(i)
    train_idx, val_idx = [], []
    for c, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    train_idx.sort(); val_idx.sort()
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """返回 (acc, macro_f1, loss)。原脚本在 autocast 下前向，此处保持一致。"""
    model.eval()
    ce = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            _, _, _, final_logits = model(images, epoch=EPOCHS)
        loss = ce(final_logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        _, predicted = final_logits.max(1)
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_labels)
    acc = 100.0 * (preds == gts).sum() / len(gts)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (gts == c)).sum()
        fp = ((preds == c) & (gts != c)).sum()
        fn = ((preds != c) & (gts == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = 100.0 * np.mean(f1s)
    return acc, macro_f1, total_loss / max(n, 1)


def train_one(seed, train_loader, val_loader, test_loader, num_classes, device):
    """单次 run：逐 epoch 记录 (test_acc, val_acc, val_macroF1, val_loss)，训练后回放三种选模规则。
       训练循环与主实验 CPGC_10种子训练.py 的 train_one 保持同源（同一损失组装、同一 optimizer/调度器）。"""
    set_seed(seed)
    model = CPGC(num_classes=num_classes).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    trace = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                proj, mlp_logits, proto_scores, final_logits = model(
                    images, labels=labels, epoch=epoch)
                loss_ce = criterion_ce(final_logits, labels)
                loss_sc = supcon_loss(proj, labels, TEMPERATURE)
                loss = loss_ce + LAMBDA_SC * loss_sc
                if proto_scores is not None:
                    loss_proto = criterion_ce(proto_scores, labels)
                    loss = loss + LAMBDA_PROTO * loss_proto
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if epoch >= PROTO_START:
                model.update_prototypes(proj.detach(), labels)
        scheduler.step()

        te_acc, _, _ = evaluate(model, test_loader, device, num_classes)
        va_acc, va_f1, va_loss = evaluate(model, val_loader, device, num_classes)
        trace.append({"epoch": epoch, "test_acc": te_acc,
                      "val_acc": va_acc, "val_macro_f1": va_f1, "val_loss": va_loss})
        if epoch % 10 == 0 or epoch == 1:
            a = torch.sigmoid(model.fusion_alpha).item()
            print(f"  [seed={seed}] Ep {epoch:>3}/{EPOCHS} α={a:.3f} "
                  f"Test {te_acc:.2f}%  ValAcc {va_acc:.2f}%  ValF1 {va_f1:.2f}%  "
                  f"ValLoss {va_loss:.3f}  ({time.time()-t0:.1f}s)")
        torch.cuda.empty_cache()

    # ---- 回放三种选模规则（同一 run，within-run 相减）----
    ep_test = max(trace, key=lambda r: r["test_acc"])["epoch"]
    ep_val = max(trace, key=lambda r: r["val_macro_f1"])["epoch"]
    ep_last = trace[-1]["epoch"]
    A_test = next(r["test_acc"] for r in trace if r["epoch"] == ep_test)
    A_val = next(r["test_acc"] for r in trace if r["epoch"] == ep_val)
    A_last = next(r["test_acc"] for r in trace if r["epoch"] == ep_last)

    del model
    torch.cuda.empty_cache()
    summary = {"seed": seed,
               "A_test": A_test, "ep_test": ep_test,
               "A_val": A_val, "ep_val": ep_val,
               "A_last": A_last, "ep_last": ep_last,
               "delta_test_minus_val": A_test - A_val,
               "delta_test_minus_last": A_test - A_last,
               "val_acc_at_val_ep": next(r["val_acc"] for r in trace if r["epoch"] == ep_val),
               "val_f1_at_val_ep": next(r["val_macro_f1"] for r in trace if r["epoch"] == ep_val)}
    return summary, trace


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}  Epochs: {EPOCHS}  ValRatio: {VAL_RATIO}")
    print(f"护栏4 判读标准（前置）: Δ<{JUDGE_THRESHOLDS['negligible']}pp 可忽略 / "
          f"{JUDGE_THRESHOLDS['negligible']}~{JUDGE_THRESHOLDS['moderate']}pp 温和 / "
          f">{JUDGE_THRESHOLDS['moderate']}pp 实质性")

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

    full_train = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=tf)
    val_eval_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=test_tf)
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
    num_classes = len(full_train.classes)
    print(f"Classes({num_classes}): {full_train.classes}  Total train imgs: {len(full_train)}")

    results, all_traces = [], []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        t0 = time.time()
        train_idx, val_idx = stratified_split(full_train, VAL_RATIO, seed)
        class_counts = [0] * num_classes
        for i in train_idx:
            class_counts[full_train.samples[i][1]] += 1
        val_counts = [0] * num_classes
        for i in val_idx:
            val_counts[full_train.samples[i][1]] += 1
        print(f"  split: train'={len(train_idx)}  val={len(val_idx)}")
        print(f"  train' per-class: {class_counts}")
        print(f"  val    per-class: {val_counts}")

        train_sub = Subset(full_train, train_idx)
        val_sub = Subset(val_eval_set, val_idx)
        if BALANCE_CLASSES:
            weights = [1.0 / class_counts[full_train.samples[i][1]] for i in train_idx]
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
        else:
            train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        res, trace = train_one(seed, train_loader, val_loader, test_loader, num_classes, device)
        results.append(res)
        for r in trace:
            r["seed"] = seed
            all_traces.append(r)
        print(f"<<< seed={seed}: A_test={res['A_test']:.2f} A_val={res['A_val']:.2f} "
              f"A_last={res['A_last']:.2f}  Δ(test-val)={res['delta_test_minus_val']:+.2f}  "
              f"({(time.time()-t0)/60:.1f} min)")

    d_test_val = np.array([r["delta_test_minus_val"] for r in results])
    d_test_last = np.array([r["delta_test_minus_last"] for r in results])
    A_test = np.array([r["A_test"] for r in results])
    A_val = np.array([r["A_val"] for r in results])
    A_last = np.array([r["A_last"] for r in results])

    print("\n" + "=" * 62)
    print("评估协议敏感性汇总（3 种子，within-run 相减；不与 10 种子并列）")
    print("=" * 62)
    print(f"A_test (test 选模) mean={A_test.mean():.2f} std={A_test.std(ddof=1):.2f}")
    print(f"A_val  (val宏F1选模) mean={A_val.mean():.2f} std={A_val.std(ddof=1):.2f}")
    print(f"A_last (末轮)        mean={A_last.mean():.2f} std={A_last.std(ddof=1):.2f}")
    print(f"Δ(test-val)  mean={d_test_val.mean():+.2f} std={d_test_val.std(ddof=1):.2f}")
    print(f"Δ(test-last) mean={d_test_last.mean():+.2f} std={d_test_last.std(ddof=1):.2f}")
    if len(d_test_val) >= 2:
        t, p = stats.ttest_rel(A_test, A_val)
        print(f"配对 t-test (A_test vs A_val): t={t:.3f}, p={p:.4f}")

    m = d_test_val.mean()
    if abs(m) < JUDGE_THRESHOLDS["negligible"]:
        verdict = "可忽略（<0.5pp）→ 可写进正文，将 best-on-test 质疑化解为卖点"
    elif abs(m) <= JUDGE_THRESHOLDS["moderate"]:
        verdict = "温和（0.5~1.5pp）→ 如实报告，说明绝对精度略偏乐观但有界"
    else:
        verdict = "实质性（>1.5pp）→ 须承认并弱化绝对精度声明，建议备而不宣"
    print(f"判读（按前置标准）: {verdict}")

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "A_test", "ep_test", "A_val", "ep_val", "A_last", "ep_last",
                    "delta_test_minus_val", "delta_test_minus_last",
                    "val_acc_at_val_ep", "val_f1_at_val_ep"])
        for r in results:
            w.writerow([r["seed"], round(r["A_test"], 2), r["ep_test"],
                        round(r["A_val"], 2), r["ep_val"], round(r["A_last"], 2), r["ep_last"],
                        round(r["delta_test_minus_val"], 2), round(r["delta_test_minus_last"], 2),
                        round(r["val_acc_at_val_ep"], 2), round(r["val_f1_at_val_ep"], 2)])
        w.writerow([])
        w.writerow(["summary", "A_test", round(A_test.mean(), 2), "±", round(A_test.std(ddof=1), 2)])
        w.writerow(["summary", "A_val", round(A_val.mean(), 2), "±", round(A_val.std(ddof=1), 2)])
        w.writerow(["summary", "A_last", round(A_last.mean(), 2), "±", round(A_last.std(ddof=1), 2)])
        w.writerow(["summary", "delta_test_minus_val", round(d_test_val.mean(), 2), "±", round(d_test_val.std(ddof=1), 2)])
        w.writerow(["summary", "delta_test_minus_last", round(d_test_last.mean(), 2), "±", round(d_test_last.std(ddof=1), 2)])
        w.writerow(["verdict", verdict])
    print(f"\n结果已保存: {CSV_PATH}")

    with open(TRACE_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "epoch", "test_acc", "val_acc", "val_macro_f1", "val_loss"])
        for r in all_traces:
            w.writerow([r["seed"], r["epoch"], round(r["test_acc"], 3),
                        round(r["val_acc"], 3), round(r["val_macro_f1"], 3), round(r["val_loss"], 4)])
    print(f"逐 epoch 轨迹已保存: {TRACE_PATH}")


if __name__ == "__main__":
    main()
