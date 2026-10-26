# -*- coding: utf-8 -*-
"""
Grad-CAM 对比可视化（对称协议重做版）
========================================
协议声明（写进图注）：
  - 模型：Baseline = ResNet-50 seed42 ; CPGC = CPGC_10seed_seed42_best.pth（真类，import 自 CPGC_10种子训练）
  - ★ 对称：两张热力图均使用「模型自由预测类别」作为 target_class（模型自己决策时看哪里）
  - 类别：crease(折痕) / rolled_pit(轧坑) / oil_spot(油斑)  —— 3 类 × 3 列（原图 / Baseline / CPGC）
  - 输入：测试集中 Baseline 预测正确的代表性样本（每类取置信度最高者）
输出：GradCAM_对称协议_3类.png (+ pdf)
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ★ 中文字体（避免方块）
import matplotlib.font_manager as _fm
for _f in [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc"]:
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
        plt.rcParams['font.sans-serif'] = [_fm.FontProperties(fname=_f).get_name()]
        break
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = "D:/EMA/GC10_dataset"
OUTPUT_DIR = SCRIPT_DIR

NUM_CLASSES = 10
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★ 只画这 3 类
TARGET_CLASSES = ['crease', 'rolled_pit', 'oil_spot']
CLASS_NAMES_CN = {'crease': '折痕', 'rolled_pit': '轧坑', 'oil_spot': '油斑'}

# 路径
CPGC_CKPT   = os.path.join(OUTPUT_DIR, "CPGC_10seed_seed42_best.pth")
BASELINE_CKPT = "D:/EMA/ResNet50基线实验/ResNet50_seed42_best.pth"

PROTO_START = 10   # 与训练脚本一致
EPOCHS = 100


# ═══════════ 模型（★ CPGC 用 import 训练真类，不再双份定义）═══════════
def build_baseline():
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    sd = torch.load(BASELINE_CKPT, map_location=DEVICE)
    m.load_state_dict(sd)
    return m.to(DEVICE).eval()


def build_cpgc():
    """★ import 训练真类（CPGC_10种子训练.py），消除重写类失同步风险"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cpgc_train", os.path.join(SCRIPT_DIR, "CPGC_10种子训练.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.CPGC(num_classes=NUM_CLASSES)
    sd = torch.load(CPGC_CKPT, map_location=DEVICE)
    msg = m.load_state_dict(sd, strict=True)   # 严格加载
    print(f"  CPGC strict load: {msg}")
    return m.to(DEVICE).eval()


class LogitsOnly(nn.Module):
    """包装真类：对外只暴露融合后的 final_logits，供 Grad-CAM 反向"""
    def __init__(self, cpgc, epoch=EPOCHS):
        super().__init__()
        self.m = cpgc
        self.epoch = epoch
    def forward(self, x):
        _, _, _, final_logits = self.m(x, epoch=self.epoch)   # ★ 走融合路径
        return final_logits


# ═══════════ Grad-CAM ═══════════
class GradCAM:
    def __init__(self, model, backbone_attr, layer):
        self.model = model
        self.gradients = None
        self.activations = None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.activations = o.detach()
    def _bwd(self, m, gi, go): self.gradients = go[0].detach()

    def generate(self, x, target_class=None):
        self.model.eval()
        x = x.to(DEVICE); x.requires_grad_(True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        if target_class is None:
            target_class = logits.argmax(1).item()
        conf = probs[0, target_class].item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits); one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0: cam = cam / cam.max()
        return cam, target_class, conf


def overlay(img_t, cam, alpha=0.5):
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img = np.clip(img_t.permute(1, 2, 0).cpu().numpy() * std + mean, 0, 1)
    hm = plt.get_cmap('jet')(cam)[:, :, :3]
    return np.clip((1 - alpha) * img + alpha * hm, 0, 1)


def main():
    print(f"Device: {DEVICE}")
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    print("[1/4] 加载模型...")
    baseline = build_baseline()
    print("  Baseline OK:", BASELINE_CKPT)
    cpgc_raw = build_cpgc()
    cpgc = LogitsOnly(cpgc_raw)          # 供 Grad-CAM
    print("  CPGC OK:", CPGC_CKPT)

    cam_bl = GradCAM(baseline, 'layer4', baseline.layer4[-1].conv3)
    cam_cp = GradCAM(cpgc, 'backbone.layer4', cpgc.m.backbone.layer4[-1].conv3)

    print("[2/4] 为每个目标类选一张 Baseline 预测正确的代表性样本...")
    test_dir = os.path.join(DATA_DIR, "test")
    all_classes = sorted(os.listdir(test_dir))
    picked = {}
    for cname in TARGET_CLASSES:
        cdir = os.path.join(test_dir, cname)
        cls_idx = all_classes.index(cname)
        best = None
        for fn in os.listdir(cdir):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')): continue
            from PIL import Image
            t = tf(Image.open(os.path.join(cdir, fn)).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p = F.softmax(baseline(t), 1)
                pred = p.argmax(1).item(); conf = p[0, pred].item()
            if pred == cls_idx and (best is None or conf > best[1]):
                best = (os.path.join(cdir, fn), conf)
        picked[cname] = best
        print(f"  {cname}: {os.path.basename(best[0]) if best else '未找到'} (conf={best[1]:.3f})" if best else f"  {cname}: 未找到")

    print("[3/4] 生成 Grad-CAM（★ 两模型均用自由预测类别，对称）...")
    nrows = len(TARGET_CLASSES)
    fig, axes = plt.subplots(nrows, 3, figsize=(9.6, 3.1 * nrows))
    log = []
    for r, cname in enumerate(TARGET_CLASSES):
        path, _ = picked[cname]
        from PIL import Image
        t = tf(Image.open(path).convert('RGB')).unsqueeze(0)
        gt = all_classes.index(cname)

        # ★ 对称协议：均 target_class=None（自由预测）
        cam_b, pred_b, conf_b = cam_bl.generate(t, target_class=None)
        cam_c, pred_c, conf_c = cam_cp.generate(t, target_class=None)

        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        orig = np.clip(t[0].permute(1, 2, 0).numpy() * std + mean, 0, 1)

        cn = CLASS_NAMES_CN[cname]
        axes[r, 0].imshow(orig); axes[r, 0].set_title(f"原图 ({cn})", fontsize=11)
        axes[r, 1].imshow(overlay(t[0], cam_b)); axes[r, 1].set_title(
            f"ResNet-50 基线  {all_classes[pred_b]}({conf_b:.2f})", fontsize=11)
        axes[r, 2].imshow(overlay(t[0], cam_c)); axes[r, 2].set_title(
            f"CPGC (本文)  {all_classes[pred_c]}({conf_c:.2f})", fontsize=11)
        for a in axes[r]: a.axis('off')

        log.append(f"{cname:11s} GT={cn:4s} | BL:{all_classes[pred_b]}({conf_b:.3f}){'✓' if pred_b==gt else '✗'} "
                   f"| CPGC:{all_classes[pred_c]}({conf_c:.3f}){'✓' if pred_c==gt else '✗'}")
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "GradCAM_对称协议_3类.png")
    out_pdf = os.path.join(OUTPUT_DIR, "GradCAM_对称协议_3类.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print("[4/4] 完成")
    print(f"  saved: {out_png}")
    print(f"  saved: {out_pdf}")
    print("\n  预测日志（对称协议 target_class=None）:")
    for l in log: print("   ", l)


if __name__ == "__main__":
    main()
