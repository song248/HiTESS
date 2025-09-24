# dataset.py
import os, numpy as np, torch
from torch.utils.data import Dataset

def build_hierarchy(n, min_len=1):
    levels = []
    L = 1
    while True:
        seg_len = n // (2**(L-1))
        if seg_len < min_len: break
        level = []
        for k in range(2**(L-1)):
            s = k*seg_len; e = min((k+1)*seg_len, n)
            level.append((s,e))
        levels.append(level)
        if seg_len==1: break
        L += 1
    return levels  # L_h, N_h는 식(2)와 일치

class VideoDataset(Dataset):
    def __init__(self, list_file, feature_root=None, modality='rgb+flow', expected_dim=2048):
        self.items=[]
        with open(list_file,'r') as f:
            for line in f:
                p,*rest = line.strip().split()
                if not p: continue
                lbl = int(rest[0]) if len(rest)>0 else -1
                nF  = int(rest[1]) if len(rest)>1 else -1
                path = p if (os.path.isabs(p) or feature_root is None) else os.path.join(feature_root, p)
                self.items.append((path,lbl,nF))
        self.modality = modality
        self.expected_dim = expected_dim

    def __len__(self): return len(self.items)

    def _load_feat(self, path):
        return np.load(path).astype(np.float32)  # (N, D)

    def __getitem__(self, idx):
        path,lbl,nF = self.items[idx]
        if self.modality=='rgb+flow':
            if path.endswith('.npy') and os.path.exists(path):  # concat 저장본
                feat = self._load_feat(path)                           # (N,2048)
            else:  # 분리 저장본 이름 규칙
                base = path.replace('_rgb.npy','').replace('_flow.npy','')
                frgb = self._load_feat(base+'_rgb.npy')
                fflw = self._load_feat(base+'_flow.npy')
                feat = np.concatenate([frgb, fflw], 1)
        elif self.modality=='rgb':
            feat = self._load_feat(path if path.endswith('_rgb.npy') or path.endswith('.npy') else path+'_rgb.npy')
        else:
            feat = self._load_feat(path if path.endswith('_flow.npy') or path.endswith('.npy') else path+'_flow.npy')

        T = feat.shape[0]
        hier = build_hierarchy(T, min_len=1)
        x   = torch.from_numpy(feat)                 # (T,D)
        y   = torch.tensor(lbl, dtype=torch.long)
        fc  = torch.tensor(nF if nF>0 else T, dtype=torch.long)
        return x, y, fc, hier
