# test.py
import argparse, json, os, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset import VideoDataset
from model import HiTESS

def load_intervals(gt_path):
    if gt_path.endswith('.json'):
        with open(gt_path,'r') as f:
            obj = json.load(f)
            return [(int(a), int(b)) for a,b in obj.get('intervals', [])]
    else:
        ints = []
        if not os.path.exists(gt_path): return ints
        with open(gt_path,'r') as f:
            for ln in f:
                sp = ln.strip().split()
                if len(sp) >= 2:
                    ints.append((int(sp[0]), int(sp[1])))
        return ints

def make_frame_labels(nF, intervals):
    y = np.zeros(nF, np.uint8)
    for s, e in intervals:
        s = max(0, s); e = min(nF, e)
        if e > s: y[s:e] = 1
    return y

def collate(batch):
    xs, ys, fcs, hiers = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(fcs), list(hiers)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_list', required=True)
    ap.add_argument('--feature_root', default=None)
    ap.add_argument('--modality', default='rgb', choices=['rgb','flow','rgb+flow'])
    ap.add_argument('--weights', required=True)
    ap.add_argument('--gt_dir', required=True, help='비디오명.json|txt (프레임 구간) 파일들이 있는 디렉토리')
    args = ap.parse_args()

    in_dim = 1024 if args.modality == 'rgb' else (2048 if args.modality == 'rgb+flow' else 1024)
    ds = VideoDataset(args.test_list, feature_root=args.feature_root, modality=args.modality,
                      expected_dim=in_dim)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HiTESS(in_dim=in_dim, hid=256, heads=4).to(dev)
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.eval()

    all_scores, all_labels = [], []
    video_scores, video_labels = [], []

    with torch.no_grad():
        for X, Y, FC, H in dl:
            X = X.to(dev)
            probs, Sfinal, _ = model(X, H)     # probs:(1,T), S:(1,)
            probs = probs.squeeze(0).cpu().numpy()  # (T,)
            nF = int(FC.item())

            # 세그먼트 → 프레임 점수 전개(균등 매핑)
            T = len(probs)
            seg_bounds = np.linspace(0, nF, T+1).astype(int)
            frame_scores = np.zeros(nF, np.float32)
            for t in range(T):
                s, e = seg_bounds[t], seg_bounds[t+1] if t < T-1 else nF
                frame_scores[s:e] = probs[t]

            # GT 로드
            name = os.path.splitext(os.path.basename(ds.items[len(video_scores)][0]))[0].replace('_rgb','').replace('_flow','')
            gt_path_json = os.path.join(args.gt_dir, name + '.json')
            gt_path_txt  = os.path.join(args.gt_dir,  name + '.txt')
            intervals = load_intervals(gt_path_json) if os.path.exists(gt_path_json) else load_intervals(gt_path_txt)
            y_frame   = make_frame_labels(nF, intervals)

            all_scores.append(frame_scores); all_labels.append(y_frame)
            video_scores.append(float(Sfinal.item()))
            video_labels.append(int(Y.item()))

    # Frame-level AUC
    fs = np.concatenate(all_scores); fl = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(fl, fs)
        print(f"Frame-level AUC: {auc*100:.2f}%")
    except Exception as e:
        print("AUC 계산 실패:", e)

    # Video-level AP
    try:
        ap = average_precision_score(np.array(video_labels), np.array(video_scores))
        print(f"Video-level AP: {ap*100:.2f}%")
    except Exception as e:
        print("AP 계산 실패:", e)

if __name__ == '__main__':
    main()
