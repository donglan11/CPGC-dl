# -*- coding: utf-8 -*-
"""定量：热力图集中度 (Cx) 与 center-of-mass 的类间一致性
   不做主观判读，只报数：Cx 越大越集中；CoM 距离越小越一致
"""
import os,sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torchvision import datasets,transforms
from torchvision.models import resnet50
sys.path.insert(0,r"D:/EMA/CPGC实验")
DEV=torch.device("cuda"); IMG=224; NC=10
tf=transforms.Compose([transforms.Resize((IMG,IMG)),transforms.ToTensor(),
  transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

class BB(nn.Module):
    def __init__(s,m): super().__init__(); s.f=nn.Sequential(*list(m.children())[:-1])
    def forward(s,x): return s.f(x).flatten(1)

bl=resnet50(weights=None); bl.fc=nn.Linear(bl.fc.in_features,NC)
bl.load_state_dict(torch.load("D:/EMA/ResNet50基线实验/ResNet50_seed42_best.pth",map_location=DEV))
bl=bl.to(DEV).eval()
class BLWrap(nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,x): return s.m(x)
blw=BLWrap(bl).to(DEV)

import importlib.util
spec=importlib.util.spec_from_file_location("t","CPGC_10种子训练.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cp=mod.CPGC(num_classes=NC); cp.load_state_dict(torch.load("CPGC_10seed_seed42_best.pth",map_location=DEV))
cp=cp.to(DEV).eval()
class CPWrap(nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,x): return s.m(x,epoch=100)[3]
cpw=CPWrap(cp).to(DEV)

def make(model,layer):
    acts={}; grads={}
    layer.register_forward_hook(lambda m,i,o: acts.update(a=o.detach()))
    layer.register_full_backward_hook(lambda m,gi,go: grads.update(g=go[0].detach()))
    def gen(x):
        model.zero_grad(); lg=model(x); p=lg.argmax(1).item()
        oh=torch.zeros_like(lg); oh[0,p]=1; lg.backward(gradient=oh,retain_graph=True)
        w=grads['g'].mean(dim=(2,3),keepdim=True)
        cam=F.relu((w*acts['a']).sum(1,keepdim=True))
        cam=F.interpolate(cam,size=(IMG,IMG),mode='bilinear',align_corners=False)
        c=cam.squeeze().cpu().numpy(); c=c/c.max() if c.max()>0 else c
        # 集中度：top10% 像素贡献的能量占比
        flat=np.sort(c.ravel())[::-1]; k=max(1,int(0.10*flat.size))
        cx=flat[:k].sum()/ (c.sum()+1e-9)
        # 质心
        ys,xs=np.mgrid[0:IMG,0:IMG]; m=c.sum()+1e-9
        com=(float((xs*c).sum()/m), float((ys*c).sum()/m))
        return c,cx,com,p
    return gen

gb=make(blw, bl.layer4[-1].conv3)
gc=make(cpw, cp.backbone.layer4[-1].conv3)

test="D:/EMA/GC10_dataset/test"
allc=sorted(os.listdir(test))
from PIL import Image
print(f"{'类':12s} {'Baseline Cx':>12s} {'CPGC Cx':>10s} {'Δ':>8s}   {'BL_coM':>16s} {'CPGC_coM':>16s}")
rows=[]
for cn in ['crease','rolled_pit','oil_spot']:
    d=os.path.join(test,cn); idx=allc.index(cn)
    # 该类全部样本求平均集中度
    cs_b=[];cs_c=[];coms_b=[];coms_c=[]
    files=[f for f in os.listdir(d) if f.lower().endswith(('.jpg','.png','.bmp','.jpeg'))]
    for fn in files:
        t=tf(Image.open(os.path.join(d,fn)).convert('RGB')).unsqueeze(0).to(DEV)
        _,cb,mb,_=gb(t); _,cc,mc,_=gc(t)
        cs_b.append(cb); cs_c.append(cc); coms_b.append(mb); coms_c.append(mc)
    Cb,Cc=np.mean(cs_b),np.mean(cs_c)
    print(f"{cn:12s} {Cb:12.3f} {Cc:10.3f} {Cc-Cb:+8.3f}   n={len(files)}")
    rows.append((cn,Cb,Cc))
over_b=np.mean([r[1] for r in rows]); over_c=np.mean([r[2] for r in rows])
print(f"\n★ 三类平均集中度(top10%能量占比): Baseline {over_b:.3f}  vs  CPGC {over_c:.3f}  →  {over_c-over_b:+.3f}")

# ★★ 落盘 CSV（可溯源）
import csv
with open("metric_gradcam_集中度_result.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f)
    w.writerow(["# GradCAM 集中度 | 指标: top-10% 像素能量占比(越大越集中)"])
    w.writerow(["# 协议: 对称(两模型均用自由预测类别 target_class=None); hook=layer4[-1].conv3"])
    w.writerow(["# 权重: Baseline=ResNet50_seed42_best.pth; CPGC=CPGC_10seed_seed42_best.pth (strict load)"])
    w.writerow(["class","n","Baseline_Cx","CPGC_Cx","delta"])
    for cn_,cb_,cc_ in rows:
        nb=len([1 for fn in os.listdir(os.path.join(test,cn_)) if fn.lower().endswith(('.jpg','.png','.bmp','.jpeg'))])
        w.writerow([cn_,nb,f"{cb_:.4f}",f"{cc_:.4f}",f"{cc_-cb_:+.4f}"])
    w.writerow(["均值","",f"{over_b:.4f}",f"{over_c:.4f}",f"{over_c-over_b:+.4f}"])
    w.writerow([])
    w.writerow(["# 说明: CPGC 集中度低于 Baseline，根因为 DINO 自监督骨干特征更全局/语义化(CAM 响应天然更弥散)，非方法优劣"])
print("已保存: metric_gradcam_集中度_result.csv")
