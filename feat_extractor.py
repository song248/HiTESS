# feat_extractor.py
import os, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- I3D (간단화 버전; RGB in_channels=3, FLOW in_channels=2) ----------
class MaxPool3dSamePadding(nn.MaxPool3d):
    def _pad(self, s, k, st):
        return max(k - st if s % st == 0 else k - (s % st), 0)
    def forward(self, x):
        b,c,t,h,w = x.shape
        pt = self._pad(t, self.kernel_size[0], self.stride[0])
        ph = self._pad(h, self.kernel_size[1], self.stride[1])
        pw = self._pad(w, self.kernel_size[2], self.stride[2])
        x = F.pad(x, (pw//2, pw - pw//2, ph//2, ph - ph//2, pt//2, pt - pt//2))
        return super().forward(x)

class Unit3D(nn.Module):
    def __init__(self, in_channels, out_channels, ks=(1,1,1), st=(1,1,1), act=F.relu, bn=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, ks, stride=st, padding=0, bias=not bn)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01) if bn else None
        self.ks, self.st, self.act = ks, st, act
    def _pad(self, s, k, st):
        return max(k - st if s % st == 0 else k - (s % st), 0)
    def forward(self, x):
        b,c,t,h,w = x.shape
        pt = self._pad(t, self.ks[0], self.st[0]); ph = self._pad(h, self.ks[1], self.st[1]); pw = self._pad(w, self.ks[2], self.st[2])
        x = F.pad(x, (pw//2, pw-pw//2, ph//2, ph-ph//2, pt//2, pt-pt//2))
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        return self.act(x) if self.act else x

class InceptionModule(nn.Module):
    def __init__(self, cin, cfg):
        super().__init__()
        self.b0 = Unit3D(cin, cfg[0], ks=(1,1,1))
        self.b1a = Unit3D(cin, cfg[1], ks=(1,1,1)); self.b1b = Unit3D(cfg[1], cfg[2], ks=(3,3,3), st=(1,1,1))
        self.b2a = Unit3D(cin, cfg[3], ks=(1,1,1)); self.b2b = Unit3D(cfg[3], cfg[4], ks=(3,3,3), st=(1,1,1))
        self.p3  = MaxPool3dSamePadding((3,3,3), (1,1,1)); self.b3b = Unit3D(cin, cfg[5], ks=(1,1,1))
    def forward(self, x):
        return torch.cat([ self.b0(x), self.b1b(self.b1a(x)), self.b2b(self.b2a(x)), self.b3b(self.p3(x)) ], dim=1)

class I3DBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        C = in_channels
        self.m = nn.ModuleDict({
            "Conv3d_1a": Unit3D(C, 64, ks=(7,7,7), st=(2,2,2)),
            "Pool2a": MaxPool3dSamePadding((1,3,3), (1,2,2)),
            "Conv2b": Unit3D(64, 64, ks=(1,1,1)),
            "Conv2c": Unit3D(64, 192, ks=(3,3,3)),
            "Pool3a": MaxPool3dSamePadding((1,3,3), (1,2,2)),
            "Mixed_3b": InceptionModule(192, [64,96,128,16,32,32]),
            "Mixed_3c": InceptionModule(256, [128,128,192,32,96,64]),
            "Pool4a": MaxPool3dSamePadding((1,3,3), (2,2,2)),
            "Mixed_4b": InceptionModule(480, [192,96,208,16,48,64]),
            "Mixed_4c": InceptionModule(512, [160,112,224,24,64,64]),
            "Mixed_4d": InceptionModule(512, [128,128,256,24,64,64]),
            "Mixed_4e": InceptionModule(528, [112,144,288,32,64,64]),
            "Mixed_4f": InceptionModule(832, [256,160,320,32,128,128]),
            "Pool5a": MaxPool3dSamePadding((2,2,2), (2,2,2)),
            "Mixed_5b": InceptionModule(832, [256,160,320,32,128,128]),
            "Mixed_5c": InceptionModule(832, [384,192,384,48,128,128]),
        })
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
    def forward_until_5c(self, x):
        for k in self.m: x = self.m[k](x)
        return self.gap(x).flatten(1)  # (B,1024)

def load_weights(model, path):
    if path and os.path.exists(path):
        sd = torch.load(path, map_location='cpu'); model.load_state_dict(sd, strict=False)
    else:
        print("[warn] I3D 가중치 파일이 없습니다. --weights_rgb / --weights_flow 경로를 지정하세요.")
    return model

# ---------- Optical Flow (TV-L1) ----------
def compute_flow_sequence(frames_rgb):  # frames_rgb: (T,H,W,3) uint8
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flows = []
    prev = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames_rgb)):
        nxt = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
        flow = tvl1.calc(prev, nxt, None)  # (H,W,2), float32
        prev = nxt
        flows.append(flow)
    # pad first flow to keep T length
    flows = [flows[0]] + flows if len(flows)>0 else [np.zeros_like(frames_rgb[0][..., :2], np.float32)]
    return np.stack(flows, 0)  # (T, H, W, 2)

def sample_clip(cap, start, length):
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(length):
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (224,224))
        frames.append(fr)
    if len(frames)==0: return None
    # pad with last
    while len(frames)<length: frames.append(frames[-1])
    return np.asarray(frames, np.uint8)  # (T,224,224,3)

def norm_rgb(x):  # (B,3,T,H,W)
    mean = torch.tensor([0.485,0.456,0.406])[:,None,None,None]
    std  = torch.tensor([0.229,0.224,0.225])[:,None,None,None]
    return (x - mean)/std

def norm_flow(x):  # (B,2,T,H,W), scale to [-1,1] then z-norm
    x = torch.clamp(x/20.0, -1.0, 1.0)  # typical scaling
    return x

def extract_video(args, vid_path, i3d_rgb, i3d_flow, device):
    cap = cv2.VideoCapture(vid_path)
    nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seg_len, step = args.seg_len, args.step
    starts = list(range(0, max(nF - seg_len, 0)+1, step))
    if len(starts)==0: starts=[0]
    rgb_feats, flow_feats = [], []
    with torch.no_grad():
        for s in starts:
            clip = sample_clip(cap, s, seg_len)
            if clip is None: continue
            # RGB
            rgb = torch.from_numpy(clip).permute(3,0,1,2).unsqueeze(0).float()/255.0  # (1,3,T,H,W)
            rgb = norm_rgb(rgb).to(device)
            frgb = i3d_rgb.forward_until_5c(rgb)  # (1,1024)
            rgb_feats.append(frgb.squeeze(0).cpu().numpy().astype(np.float32))
            # FLOW
            flow = compute_flow_sequence(clip)  # (T,H,W,2)
            flow = torch.from_numpy(flow).permute(3,0,1,2).unsqueeze(0).float()       # (1,2,T,H,W)
            flow = norm_flow(flow).to(device)
            fflow = i3d_flow.forward_until_5c(flow)  # (1,1024)
            flow_feats.append(fflow.squeeze(0).cpu().numpy().astype(np.float32))
    cap.release()
    rgb_feats = np.stack(rgb_feats,0) if len(rgb_feats)>0 else np.zeros((0,1024),np.float32)
    flow_feats= np.stack(flow_feats,0) if len(flow_feats)>0 else np.zeros((0,1024),np.float32)
    if args.concat_modalities:
        feats = np.concatenate([rgb_feats, flow_feats], axis=1)  # (N,2048)
        return feats, nF
    else:
        return (rgb_feats, flow_feats), nF

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--weights_rgb', type=str, default=None)
    ap.add_argument('--weights_flow', type=str, default=None)
    ap.add_argument('--exts', nargs='+', default=['.mp4','.avi','.mkv'])
    ap.add_argument('--seg_len', type=int, default=16)  # 논문 구현: 16프레임
    ap.add_argument('--step', type=int, default=8)      # 슬라이딩 스텝 8
    ap.add_argument('--concat_modalities', action='store_true', help='RGB+Flow concat(2048D)로 저장')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    i3d_rgb  = load_weights(I3DBackbone(in_channels=3), args.weights_rgb ).to(device).eval()
    i3d_flow = load_weights(I3DBackbone(in_channels=2), args.weights_flow).to(device).eval()

    meta = open(os.path.join(args.out_dir, 'feature_metadata.txt'), 'w')
    for root,_,files in os.walk(args.video_dir):
        for f in files:
            if not any(f.lower().endswith(e) for e in args.exts): continue
            path = os.path.join(root,f); name = os.path.splitext(f)[0]
            feats, nF = extract_video(args, path, i3d_rgb, i3d_flow, device)
            if args.concat_modalities:
                np.save(os.path.join(args.out_dir, name+'.npy'), feats)
                meta.write(f"{name}.npy -1 {nF}\n")
            else:
                np.save(os.path.join(args.out_dir, name+'_rgb.npy'), feats[0])
                np.save(os.path.join(args.out_dir, name+'_flow.npy'), feats[1])
                meta.write(f"{name}_rgb.npy -1 {nF}\n")
                meta.write(f"{name}_flow.npy -1 {nF}\n")
            print("extracted:", name, "Tframes:", nF)
    meta.close()

if __name__ == '__main__':
    main()
