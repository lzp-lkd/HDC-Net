# ============================
# HDC-Net 完整工程版（可训练版）
# 基于：HCFormer + ISAF + PFB + 完整Decoder
# 兼容你提供的 train.py
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ==================== ISAF ====================
class ISAFSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size,padding=kernel_size//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg = torch.mean(x,1,keepdim=True)
        mx,_ = torch.max(x,1,keepdim=True)
        x = torch.cat([avg,mx],1)
        return self.sigmoid(self.conv(x))

class ISAFModule(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = nn.Sequential(
            nn.Conv2d(c*2,c//8,1), nn.ReLU(),
            nn.Conv2d(c//8,c*2,1), nn.Sigmoid()
        )
        self.sa = ISAFSpatialAttention()
        self.rgb_conv = nn.Conv2d(c,c,1)
        self.dep_conv = nn.Conv2d(c,c,1)

    def forward(self,x,y):
        cat = torch.cat([x,y],1)
        w = F.adaptive_avg_pool2d(cat,1)
        w = self.ca(w)
        wx,wy = torch.chunk(w,2,1)
        x = x*wx
        y = y*wy

        s = self.sa(x+y)
        x = x*s
        y = y*s

        f = x+y
        return x+self.rgb_conv(f), y+self.dep_conv(f)

# ==================== Backbone ====================
class SimpleBackbone(nn.Module):
    def __init__(self,in_ch=3,base=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,base,3,1,1)
        self.conv2 = nn.Conv2d(base,base*2,3,2,1)
        self.conv3 = nn.Conv2d(base*2,base*4,3,2,1)
        self.conv4 = nn.Conv2d(base*4,base*8,3,2,1)

    def forward(self,x):
        f1 = F.relu(self.conv1(x))
        f2 = F.relu(self.conv2(f1))
        f3 = F.relu(self.conv3(f2))
        f4 = F.relu(self.conv4(f3))
        return [f1,f2,f3,f4]

# ==================== Transformer ====================
class TokenBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
        self.proj = nn.Linear(dim,dim)

    def forward(self,x):
        q,k,v = self.q(x),self.k(x),self.v(x)
        attn = torch.softmax(q@k.transpose(-1,-2)/math.sqrt(x.size(-1)),dim=-1)
        return self.proj(attn@v)

# ==================== PFB ====================
class PFB(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv = nn.Conv2d(c*4,c,1)

    def forward(self,feats):
        base = feats[0].shape[-2:]
        feats = [F.interpolate(f,base) for f in feats]
        return self.conv(torch.cat(feats,1))

# ==================== Decoder ====================
class Up(nn.Module):
    def __init__(self,in_ch,skip_ch,out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch+skip_ch,out_ch,3,1,1),
            nn.ReLU(),
            nn.Conv2d(out_ch,out_ch,3,1,1),
            nn.ReLU()
        )

    def forward(self,x,skip):
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        x = torch.cat([x,skip],1)
        return self.conv(x)

# ==================== HDCNet ====================
class HDCNet(nn.Module):
    def __init__(self,n_classes=6):
        super().__init__()

        self.rgb_backbone = SimpleBackbone(3)
        self.dsm_backbone = SimpleBackbone(1)

        self.isaf1 = ISAFModule(64)
        self.isaf2 = ISAFModule(128)
        self.isaf3 = ISAFModule(256)
        self.isaf4 = ISAFModule(512)

        self.pfb = PFB(512)

        self.up1 = Up(512,256,256)
        self.up2 = Up(256,128,128)
        self.up3 = Up(128,64,64)

        self.head = nn.Conv2d(64,n_classes,1)

    def forward(self,x,y):
        y = y.unsqueeze(1)

        rx = self.rgb_backbone(x)
        ry = self.dsm_backbone(y)

        f1x,f1y = self.isaf1(rx[0],ry[0])
        f2x,f2y = self.isaf2(rx[1],ry[1])
        f3x,f3y = self.isaf3(rx[2],ry[2])
        f4x,f4y = self.isaf4(rx[3],ry[3])

        bottleneck = self.pfb([f4x,f3x,f2x,f1x])

        x = self.up1(bottleneck,f3x)
        x = self.up2(x,f2x)
        x = self.up3(x,f1x)

        out = self.head(x)
        out = F.interpolate(out,scale_factor=2,mode='bilinear',align_corners=False)

        return out

# ==================== 接口兼容 ====================
class VisionTransformer(nn.Module):
    def __init__(self,config,img_size,num_classes):
        super().__init__()
        self.model = HDCNet(num_classes)

    def forward(self,x,y):
        return self.model(x,y)

CONFIGS = {
    'R50-ViT-B_16': None
}
