# model.py
import torch
import torch.nn as nn

class HiTESS(nn.Module):
    """
    BiLSTM + Multi-Head Attention -> segment-wise probs
    + Hierarchical scoring (Algorithm 1) with AVERAGE pooling:
      - 각 (레벨, 세그먼트구간)마다 서브시퀀스를 인코딩(파라미터 공유)
      - 구간 점수들을 '레벨 평균'으로 요약
      - 최종 점수 S_final = 레벨평균들의 평균 (Avg 고정, 논문 Eq.(8))
    """
    def __init__(self, in_dim=1024, hid=256, heads=4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=2*hid, num_heads=heads, batch_first=True)
        self.ln   = nn.LayerNorm(2*hid)
        self.head = nn.Linear(2*hid, 1)   # segment-wise logit
        self.sigmoid = nn.Sigmoid()

    # ----- 공통 인코딩: 임의 길이의 시퀀스 x_seq (T,D) -----
    def encode_sequence(self, x_seq):
        """
        x_seq: (T,D)
        return:
          seg_probs: (T,) in [0,1]
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)  # (1,T,D)
        h, _ = self.lstm(x_seq)                 # (1,T,2H)
        a, _ = self.attn(h, h, h)               # (1,T,2H)
        e    = self.ln(h + a)                   # (1,T,2H)
        logits = self.head(e).squeeze(-1)       # (1,T)
        probs  = self.sigmoid(logits).squeeze(0)  # (T,)
        return probs

    # ----- 계층 스코어링 (Avg 고정 + 서브시퀀스 재인코딩) -----
    def hierarchical_score(self, x_seq, hierarchy):
        """
        x_seq     : (T,D)
        hierarchy : [ [(s,e), ...],   # level 1
                      [(s,e), ...],   # level 2
                      ...
                    ]
        return:
          S_final      : scalar
          level_scores : list of tensors, each (num_segments_in_level,)
        """
        device = x_seq.device
        level_scores = []
        level_means  = []

        for level in hierarchy:
            seg_scores = []
            for (s, e) in level:
                # 서브시퀀스 인코딩 (Algorithm 1: 서브시퀀스에도 동일 인코더 적용)
                seg_probs = self.encode_sequence(x_seq[s:e])     # (len,)
                seg_score = seg_probs.mean()                     # 구간 평균 (Avg 고정)
                seg_scores.append(seg_score)
            if len(seg_scores) == 0:
                lv = torch.zeros(1, device=device)
            else:
                lv = torch.stack(seg_scores)                     # (N_level,)
            level_scores.append(lv)
            level_means.append(lv.mean())                        # 레벨 평균

        # 최종 점수 = 레벨 평균들의 평균 (레벨 수에 균등 가중)
        if len(level_means) == 0:
            S_final = torch.zeros((), device=device)
        else:
            S_final = torch.stack(level_means).mean()

        return S_final, level_scores

    def forward(self, x, hierarchy):
        """
        x         : (B,T,D)
        hierarchy : python list len=B, 각 원소는 위의 'levels' 포맷
        return:
          seg_probs   : (B,T)  # 전체 시퀀스(원본)에 대한 세그먼트 확률
          S_final_all : (B,)   # 계층 전역 점수
          level_scores: list(len=B) of list(level_tensors)
        """
        B, T, D = x.shape

        # 전체 시퀀스에 대해 한 번 인코딩 (로컬라이제이션/시각화에 사용)
        seg_probs_list = []
        for b in range(B):
            seg_probs_list.append(self.encode_sequence(x[b]))
        seg_probs = torch.stack(seg_probs_list, dim=0)  # (B,T)

        # 계층 스코어 (서브시퀀스 재인코딩)
        S_list, Lscores = [], []
        for b in range(B):
            Sf, lvs = self.hierarchical_score(x[b], hierarchy[b])
            S_list.append(Sf)
            Lscores.append(lvs)
        S_final_all = torch.stack(S_list, dim=0)  # (B,)

        return seg_probs, S_final_all, Lscores
