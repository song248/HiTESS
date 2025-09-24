# train.py
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import VideoDataset
from model import HiTESS

# --------- MIL Ranking Loss (Eq.(9)) ----------
def mil_ranking_loss(S, y, margin=1.0):
    """
    S: (B,) global hierarchical scores
    y: (B,) 1=anomaly, 0=normal
    """
    pos = (y == 1).nonzero(as_tuple=True)[0]
    neg = (y == 0).nonzero(as_tuple=True)[0]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0., device=S.device)
    Sp = S[pos][:, None]   # (P,1)
    Sn = S[neg][None, :]   # (1,N)
    loss = torch.relu(margin - (Sp - Sn)).mean()
    return loss

# --------- Weighted BCE (Sec.3.5) ----------
def weighted_bce(seg_probs, video_labels, w_pos=1.0, w_neg=1.0, eps=1e-6):
    """
    seg_probs   : (B,T) in [0,1]
    video_labels: (B,) 0/1  (weak supervision -> 모든 세그먼트에 브로드캐스트)
    """
    B, T = seg_probs.shape
    y = video_labels.float().unsqueeze(1).expand(B, T)  # (B,T)
    p = seg_probs.clamp(min=eps, max=1.0 - eps)
    # -[ w_pos * y * log(p) + w_neg * (1-y) * log(1-p) ]
    loss = -(w_pos * y * torch.log(p) + w_neg * (1.0 - y) * torch.log(1.0 - p)).mean()
    return loss

def collate(batch):
    xs, ys, fcs, hiers = zip(*batch)
    X  = torch.stack(xs)      # (B,T,D)  (길이 통일된 feature 기준)
    Y  = torch.stack(ys)      # (B,)
    FC = torch.stack(fcs)     # (B,)
    return X, Y, FC, list(hiers)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_list', required=True)
    ap.add_argument('--feature_root', default=None)
    # 재현 수치 일관성: 기본 모달리티를 I3D-RGB로
    ap.add_argument('--modality', default='rgb', choices=['rgb','flow','rgb+flow'])
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--save', default='checkpoints/hitess.pth')

    # Loss 하이퍼파라미터
    ap.add_argument('--margin', type=float, default=1.0)     # MIL hinge margin
    ap.add_argument('--lambda_rank', type=float, default=1.0)
    ap.add_argument('--lambda_bce',  type=float, default=0.5)
    ap.add_argument('--w_pos', type=float, default=1.0)      # BCE class weights
    ap.add_argument('--w_neg', type=float, default=1.0)
    args = ap.parse_args()

    in_dim = 1024 if args.modality == 'rgb' else (2048 if args.modality == 'rgb+flow' else 1024)
    ds = VideoDataset(args.train_list, feature_root=args.feature_root,
                      modality=args.modality, expected_dim=in_dim)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HiTESS(in_dim=in_dim, hid=256, heads=args.heads).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, Y, _, H in dl:
            X, Y = X.to(dev), Y.to(dev)
            opt.zero_grad()
            seg_probs, Sfinal, _ = model(X, H)  # seg_probs:(B,T), Sfinal:(B,)

            L_rank = mil_ranking_loss(Sfinal, Y, margin=args.margin)
            L_bce  = weighted_bce(seg_probs, Y, w_pos=args.w_pos, w_neg=args.w_neg)
            loss   = args.lambda_rank * L_rank + args.lambda_bce * L_bce

            loss.backward()
            opt.step()
            total += float(loss.item())

        print(f"[{ep}/{args.epochs}] total={total/len(dl):.4f}")

    torch.save(model.state_dict(), args.save)
    print("saved:", args.save)

if __name__ == '__main__':
    main()
