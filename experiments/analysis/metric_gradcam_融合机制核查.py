# -*- coding: utf-8 -*-
"""核查：CPGC 的 Grad-CAM 弥散是否与其最终决策=MLP+原型融合有关
   实验：对 CPGC 分别用 final_logits（融合）与 mlp_logits（纯MLP分支）做 Grad-CAM
"""
import os,sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torchvision import transforms
sys.path.insert(0,r"D:/EMA/CPGC实验")
DEV=torch.device("cuda"); IMG=224; NC=10
tf=transforms.Compose([transforms.Resize((IMG,IMG)),transforms.ToTensor(),
  transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
import importlib.util
spec=importlib.util.spec_from_file_location("t","CPGC_10种子训练.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cp=mod.CPGC(num_classes=NC); cp.load_state_dict(torch.load("CPGC_10seed_seed42_best.pth",map_location=DEV))
cp=cp.to(DEV).eval()
print("fusion_alpha(sigmoid后)=%.4f"%torch.sigmoid(cp.fusion_alpha).item())

class W(nn.Module):
    def __init__(s,m,mode): super().__init__(); s.m=m; s.mode=mode
    def forward(s,x):
        proj,mlp,proto,fin=s.m(x,epoch=100)
        return fin if s.mode=='final' else mlp

def make(model,layer):
    st={}
    layer.register_forward_hook(lambda m,i,o: st.update(a=o.detach()))
    layer.register_full_backward_hook(lambda m,gi,go: st.update(g=go[0].detach()))
    def gen(x):
        model.zero_grad(); lg=model(x); p=lg.argmax(1).item()
        oh=torch.zeros_like(lg); oh[0,p]=1; lg.backward(gradient=oh,retain_graph=True)
        w=st['g'].mean(dim=(2,3),keepdim=True)
        cam=F.relu((w*st['a']).sum(1,keepdim=True))
        cam=F.interpolate(cam,size=(IMG,IMG),mode='bilinear',align_corners=False)
        c=cam.squeeze().cpu().numpy(); c=c/c.max() if c.max()>0 else c
        flat=np.sort(c.ravel())[::-1]; k=int(0.10*flat.size)
        return flat[:k].sum()/(c.sum()+1e-9)
    return gen

from PIL import Image
test="D:/EMA/GC10_dataset/test"
for mode,label in [('final','CPGC final_logits(融合)'),('mlp','CPGC mlp_logits(纯判别分支)')]:
    g=make(W(cp,mode).to(DEV), cp.backbone.layer4[-1].conv3)
    vals=[]
    for cn in ['crease','rolled_pit','oil_spot']:
        d=os.path.join(test,cn)
        for fn in os.listdir(d):
            if fn.lower().endswith(('.jpg','.png','.bmp','.jpeg')):
                t=tf(Image.open(os.path.join(d,fn)).convert('RGB')).unsqueeze(0).to(DEV)
                vals.append(g(t))
    print(f"  {label:26s} 集中度均值={np.mean(vals):.3f}")
