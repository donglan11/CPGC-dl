"""
Grad-CAM 可视化：Baseline vs CPGC 对比
每类选1张代表性图片，生成原图 + Baseline热力图 + CPGC热力图 的对比图
输出：10类 × 3列 的大对比图 + 每类单独的小图
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# ─── 路径 ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "D:/EMA/GC10_dataset"
OUTPUT_DIR = SCRIPT_DIR
sys.path.insert(0, SCRIPT_DIR)

# ─── 常量 ───
NUM_CLASSES = 10
IMG_SIZE = 224
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 类名 ───
CLASS_NAMES_CN = {
    'crease': '折痕',
    'crescent_gap': '月牙缺口',
    'inclusion': '夹杂',
    'oil_spot': '油斑',
    'punching_hole': '冲孔',
    'rolled_pit': '轧坑',
    'silk_spot': '丝斑',
    'waist folding': '腰折',
    'water_spot': '水斑',
    'welding_line': '焊线',
}

# ═══════════════════════════════════════════════
# 模型定义 (与训练脚本一致)
# ═══════════════════════════════════════════════

class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


DINO_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "dino_resnet50_pretrain.pth")
DINO_WEIGHTS_URLS = [
    "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
]

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
    backbone = resnet50()
    backbone.fc = nn.Identity()
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f"DINO backbone loaded: {msg}")
    return backbone


class CPGC(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, proj_dim=128):
        super().__init__()
        self.backbone = load_dino_backbone()
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
        self.register_buffer("prototypes", torch.randn(num_classes, proj_dim) * 0.01)
        self.register_buffer("proto_count", torch.zeros(num_classes))
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.log_proto_temp = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, x):
        feat = self.backbone(x)
        proj = self.projection(feat)
        mlp_logits = self.classifier(feat)
        proto_norm = F.normalize(self.prototypes, dim=1)
        proj_norm = F.normalize(proj, dim=1)
        proto_scores = proj_norm @ proto_norm.T
        temp = torch.exp(self.log_proto_temp).clamp(min=0.01)
        proto_scores = proto_scores / temp
        alpha = torch.sigmoid(self.fusion_alpha)
        final_logits = alpha * mlp_logits + (1 - alpha) * proto_scores
        return final_logits


# ═══════════════════════════════════════════════
# Grad-CAM 核心
# ═══════════════════════════════════════════════

class GradCAM:
    """
    Grad-CAM: 对 ResNet-50 backbone 的最后一个卷积层 (layer4[-1].conv3)
    计算梯度加权激活图
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None

        if target_layer is None:
            # ResNet-50 最后一个卷积层
            target_layer = model.backbone.layer4[-1].conv3

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        生成 Grad-CAM 热力图
        input_tensor: (1, 3, 224, 224)
        target_class: int, 目标类别索引; None 则用模型预测类别
        返回: (heatmap, pred_class, pred_conf)
            heatmap: (224, 224) numpy array, 0~1
        """
        self.model.eval()
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        # 前向
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        conf = probs[0, target_class].item()

        # 反向
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM 计算
        # gradients: (1, C, H, W)
        # activations: (1, C, H, W)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # 上采样到输入尺寸
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # 归一化到 0~1
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, conf


# ═══════════════════════════════════════════════
# 选图：每类选1张预测正确的图片
# ═══════════════════════════════════════════════

def select_representative_images(model, test_set, class_names, n_per_class=1):
    """
    每类选 n_per_class 张模型预测正确的图片
    优先选置信度高的 (更有说服力)
    """
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    selected = {cn: [] for cn in class_names}

    # 遍历每类
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(test_set, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        candidates = []
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            fpath = os.path.join(cls_dir, fname)
            try:
                from PIL import Image
                img = Image.open(fpath).convert('RGB')
                tensor = tf(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = F.softmax(logits, dim=1)
                    pred = probs.argmax(dim=1).item()
                    conf = probs[0, pred].item()

                if pred == cls_idx:
                    candidates.append((fpath, conf))
            except Exception:
                continue

        # 按置信度降序，选前 n_per_class 张
        candidates.sort(key=lambda x: -x[1])
        for fpath, conf in candidates[:n_per_class]:
            selected[cls_name].append((fpath, conf))

    return selected


# ═══════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════

def apply_colormap(gray_cam, colormap='jet', alpha=0.5):
    """将灰度热力图转为彩色叠加图"""
    cm = plt.get_cmap(colormap)
    colored = cm(gray_cam)[:, :, :3]  # 去掉 alpha 通道
    return colored


def show_cam_on_image(img_tensor, cam, alpha=0.5):
    """
    将 Grad-CAM 叠加到原图上
    img_tensor: (3, 224, 224) 已归一化的 tensor
    cam: (224, 224) 0~1
    返回: (224, 224, 3) uint8
    """
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # 热力图上色
    heatmap = apply_colormap(cam, 'jet', alpha)

    # 叠加
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ═══════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")

    # ─── 加载模型 ───
    print("\n[1/5] 加载 Baseline 模型...")
    baseline = ResNet50Baseline(NUM_CLASSES).to(DEVICE)
    bl_path = "D:/EMA/ResNet50基线实验/ResNet50_seed42_best.pth"
    bl_state = torch.load(bl_path, map_location=DEVICE)
    # 原始权重 key 无前缀 (conv1.weight, layer1.0.conv1.weight...)
    # ResNet50Baseline 的 backbone 嵌套了一层，需要加 backbone. 前缀
    bl_state_new = {}
    for k, v in bl_state.items():
        if k == 'fc.weight':
            bl_state_new['classifier.weight'] = v
        elif k == 'fc.bias':
            bl_state_new['classifier.bias'] = v
        else:
            bl_state_new['backbone.' + k] = v
    baseline.load_state_dict(bl_state_new)
    baseline.eval()
    print(f"  权重: {bl_path}")

    print("[2/5] 加载 CPGC 模型...")
    cpgc = CPGC(NUM_CLASSES, proj_dim=128).to(DEVICE)
    cpgc_path = os.path.join(OUTPUT_DIR, "CPGC_10seed_seed42_best.pth")
    cpgc.load_state_dict(torch.load(cpgc_path, map_location=DEVICE), strict=False)
    cpgc.eval()
    print(f"  权重: {cpgc_path}")

    # ─── Grad-CAM 钩子 ───
    print("\n[3/5] 初始化 Grad-CAM...")
    cam_baseline = GradCAM(baseline)
    cam_cpgc = GradCAM(cpgc)

    # ─── 选图 ───
    print("\n[4/5] 每类选取代表性图片 (Baseline预测正确的)...")
    test_dir = os.path.join(DATA_DIR, "test")
    class_names = sorted(os.listdir(test_dir))
    print(f"  类别: {class_names}")

    selected = select_representative_images(baseline, test_dir, class_names, n_per_class=1)
    for cn in class_names:
        if selected[cn]:
            print(f"  {cn}: {os.path.basename(selected[cn][0][0])} (conf={selected[cn][0][1]:.3f})")
        else:
            print(f"  {cn}: 未找到预测正确的图片")

    # ─── 生成 Grad-CAM 对比图 ───
    print("\n[5/5] 生成 Grad-CAM 热力图...")

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 大对比图: 10行 × 5列 (原图 | Baseline预测 | Baseline热力图 | CPGC预测 | CPGC热力图)
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 5, figsize=(20, 2.8 * n_classes))
    fig.suptitle('Grad-CAM 可视化对比: Baseline vs CPGC', fontsize=18, fontweight='bold', y=0.995)

    results_log = []
    bl_correct_count = 0
    cp_correct_count = 0

    for row, cls_name in enumerate(class_names):
        if not selected[cls_name]:
            continue

        img_path = selected[cls_name][0][0]
        from PIL import Image
        pil_img = Image.open(img_path).convert('RGB')
        tensor = tf(pil_img).unsqueeze(0)

        # 原图 (反归一化)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        orig_img = tensor[0].permute(1, 2, 0).numpy() * std + mean
        orig_img = np.clip(orig_img, 0, 1)

        # Baseline Grad-CAM
        cam_bl, pred_bl, conf_bl = cam_baseline.generate(tensor, target_class=None)
        overlay_bl = show_cam_on_image(tensor[0], cam_bl, alpha=0.5)

        # CPGC Grad-CAM (使用 ground truth 类别，确保对比公平)
        gt_class = class_names.index(cls_name)
        cam_cp, pred_cp, conf_cp = cam_cpgc.generate(tensor, target_class=gt_class)
        overlay_cp = show_cam_on_image(tensor[0], cam_cp, alpha=0.5)

        # 也生成 CPGC 自由预测的 Grad-CAM
        cam_cp_free, pred_cp_free, conf_cp_free = cam_cpgc.generate(tensor, target_class=None)

        pred_bl_name = class_names[pred_bl] if pred_bl < len(class_names) else '?'
        pred_cp_name = class_names[pred_cp] if pred_cp < len(class_names) else '?'
        gt_cn = CLASS_NAMES_CN.get(cls_name, cls_name)

        # 记录日志
        bl_hit = '[OK]' if pred_bl == gt_class else '[NO]'
        cp_hit = '[OK]' if pred_cp == gt_class else '[NO]'
        if pred_bl == gt_class:
            bl_correct_count += 1
        if pred_cp == gt_class:
            cp_correct_count += 1
        results_log.append(f"{cls_name:15s} | GT={gt_cn} | "
                          f"BL: {pred_bl_name}({conf_bl:.3f}) [{bl_hit}] | "
                          f"CPGC: {pred_cp_name}({conf_cp:.3f}) [{cp_hit}]")

        # ─── 画子图 ───
        # 列0: 原图
        axes[row, 0].imshow(orig_img)
        axes[row, 0].set_title(f'原图\n{gt_cn}', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        # 列1: Baseline 纯热力图
        axes[row, 1].imshow(cam_bl, cmap='jet', vmin=0, vmax=1)
        axes[row, 1].set_title(f'Baseline 注意力', fontsize=11)
        axes[row, 1].axis('off')

        # 列2: Baseline 叠加
        axes[row, 2].imshow(overlay_bl)
        correct_bl = 'OK' if pred_bl == gt_class else 'NO'
        axes[row, 2].set_title(f'Baseline 叠加 [{correct_bl}]\n{pred_bl_name} {conf_bl:.2f}', fontsize=11)
        axes[row, 2].axis('off')

        # 列3: CPGC 纯热力图
        axes[row, 3].imshow(cam_cp, cmap='jet', vmin=0, vmax=1)
        axes[row, 3].set_title(f'CPGC 注意力', fontsize=11)
        axes[row, 3].axis('off')

        # 列4: CPGC 叠加
        axes[row, 4].imshow(overlay_cp)
        correct_cp = 'OK' if pred_cp == gt_class else 'NO'
        axes[row, 4].set_title(f'CPGC 叠加 [{correct_cp}]\n{pred_cp_name} {conf_cp:.2f}', fontsize=11)
        axes[row, 4].axis('off')

        # 正确/错误用颜色区分行标题
        row_color = '#2E7D32' if pred_bl == gt_class else '#C62828'

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # ─── 保存 ───
    out_path = os.path.join(OUTPUT_DIR, 'GradCAM_对比图.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n已保存: {out_path}")

    # 打印日志
    print(f"\n{'='*70}")
    print("  Grad-CAM 预测对比日志")
    print(f"{'='*70}")
    for line in results_log:
        print(f"  {line}")

    # 统计
    print(f"\n  Baseline 正确: {bl_correct_count}/{n_classes}")
    print(f"  CPGC     正确: {cp_correct_count}/{n_classes}")
    print("完成!")


if __name__ == "__main__":
    main()
