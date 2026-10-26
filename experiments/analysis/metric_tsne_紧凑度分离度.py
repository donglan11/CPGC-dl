# -*- coding: utf-8 -*-
"""t-SNE 定量佐证：类内紧凑度 / 类间分离度（在 t-SNE 2D 嵌入上）
   不重跑 t-SNE，直接复用脚本同一流程与同一 seed，保证与图一致。
"""
import os, sys, numpy as np, torch, torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
sys.path.insert(0, r"D:/EMA/CPGC实验")

DATA_DIR="D:/EMA/GC10_dataset"; NUM_CLASSES=10; IMG_SIZE=224
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
ts=datasets.ImageFolder(os.path.join(DATA_DIR,"test"),transform=tf)
tl=DataLoader(ts,batch_size=32,shuffle=False,num_workers=0)
cn=ts.classes

def feats(model,loader):
    model.eval(); F=[]; L=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(dev)
            with autocast(): f=model(x)
            F.append(f.float().cpu().numpy()); L.append(y.numpy())
    return np.concatenate(F), np.concatenate(L)

from CPGC_3种子训练 import CPGC as CM
cm=CM(num_classes=NUM_CLASSES).to(dev)
cm.load_state_dict(torch.load("CPGC_10seed_seed42_best.pth",map_location=dev))
class BB(nn.Module):
    def __init__(s,m): super().__init__(); s.b=m.backbone
    def forward(s,x): return s.b(x)
cf,lab=feats(BB(cm).to(dev),tl)

from torchvision.models import resnet50
bm=resnet50(weights=None); bm.fc=nn.Linear(bm.fc.in_features,NUM_CLASSES)
bm.load_state_dict(torch.load("D:/EMA/ResNet50基线实验/ResNet50_seed42_best.pth",map_location=dev))
bm=bm.to(dev)
class BB2(nn.Module):
    def __init__(s,m): super().__init__(); s.f=nn.Sequential(*list(m.children())[:-1])
    def forward(s,x): return s.f(x).flatten(1)
bf,_=feats(BB2(bm).to(dev),tl)
print("feat shapes:", bf.shape, cf.shape)

# ★ 与脚本完全一致的联合降维
allf=np.vstack([bf,cf])
emb=TSNE(n_components=2,perplexity=30,n_iter=1000,random_state=42).fit_transform(allf)
n=len(lab)
E_bl, E_cp = emb[:n], emb[n:]

def metrics(E, y, name):
    cents=np.array([E[y==c].mean(0) for c in range(NUM_CLASSES)])
    intra=[]
    for c in range(NUM_CLASSES):
        P=E[y==c]; intra.append(np.sqrt(((P-cents[c])**2).sum(1)).mean())
    intra=np.array(intra)
    # 类间：各质心两两距离
    from itertools import combinations
    inter=np.array([np.linalg.norm(cents[i]-cents[j]) for i,j in combinations(range(NUM_CLASSES),2)])
    sep=inter.mean()/intra.mean()          # 分离度：越大越好
    print(f"\n[{name}]")
    print(f"  类内平均散布(越小越紧凑) : {intra.mean():.3f}")
    print(f"  类间质心平均距离(越大越好): {inter.mean():.3f}")
    print(f"  ★ 分离度 inter/intra     : {sep:.3f}")
    # 逐类紧凑度（关注 crease / rolled_pit / silk_spot）
    for name_c in ['crease','rolled_pit','silk_spot']:
        i=cn.index(name_c)
        print(f"    - {name_c:11s} 类内散布={intra[i]:.3f}  n={int((y==i).sum())}")
    return sep, intra.mean(), inter.mean(), intra

s_bl=metrics(E_bl,lab,"Baseline")
s_cp=metrics(E_cp,lab,"CPGC")
print(f"\n★★ 分离度提升: {s_cp[0]:.3f} vs {s_bl[0]:.3f}  → {(s_cp[0]/s_bl[0]-1)*100:+.1f}%")
print(f"★★ 类内散布变化: {s_cp[1]:.3f} vs {s_bl[1]:.3f}  → {(s_cp[1]/s_bl[1]-1)*100:+.1f}%")
print(f"★★ 类间质心距变化: {s_cp[2]:.3f} vs {s_bl[2]:.3f}  → {(s_cp[2]/s_bl[2]-1)*100:+.1f}%   ← 必报(千问)")

# ★★ 落盘 CSV（可溯源）
import csv
with open("metric_tsne_result.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f)
    w.writerow(["# tSNE 定量指标 | 协议: vstack联合降维, perplexity=30, random_state=42, n_iter=1000"])
    w.writerow(["# 特征: backbone 2048-d, GC10 test 466张; CPGC=CPGC_10seed_seed42_best.pth, Baseline=ResNet50_seed42_best.pth"])
    w.writerow(["method","intra_mean(类内越小越紧凑)","inter_mean(类间质心距越大越好)","separability(inter/intra,越大越好)"])
    w.writerow(["Baseline",f"{s_bl[1]:.4f}",f"{s_bl[2]:.4f}",f"{s_bl[0]:.4f}"])
    w.writerow(["CPGC",f"{s_cp[1]:.4f}",f"{s_cp[2]:.4f}",f"{s_cp[0]:.4f}"])
    w.writerow(["变化%",f"{(s_cp[1]/s_bl[1]-1)*100:+.1f}",f"{(s_cp[2]/s_bl[2]-1)*100:+.1f}",f"{(s_cp[0]/s_bl[0]-1)*100:+.1f}"])
    w.writerow([])
    w.writerow(["# 逐类类内散布"])
    w.writerow(["class","n","Baseline_intra","CPGC_intra","变化%"])
    for i,c in enumerate(cn):
        nb=int((lab==i).sum())
        w.writerow([c,nb,f"{s_bl[3][i]:.4f}",f"{s_cp[3][i]:.4f}",f"{(s_cp[3][i]/s_bl[3][i]-1)*100:+.1f}"])
print("\n已保存: metric_tsne_result.csv")
