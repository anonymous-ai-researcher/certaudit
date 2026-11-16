"""Encoders of the paper (unchanged math) plus a transformer variant, all device-aware.

  AIGEncoder      DeepGate-style level-wise propagation, GRU updates, R rounds (the paper's encoder)
  AIGEncoderBi    + attention-weighted fanin aggregation and a reverse (fanout) pass (the paper's "Bi")
  AIGTransformer  AIGEncoder followed by L global self-attention layers over the gates of each circuit,
                  with a learned logic-level embedding (new in the GPU study; NOT DeepGate3 itself)"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def collate(samples):
    """Disjoint union of samples. Returns dict of tensors + per-sample node offsets."""
    offs, N = [], 0
    fa, fb, ia, ib, gid, typ, prob, lev = [], [], [], [], [], [], [], []
    for s in samples:
        offs.append(N)
        n = 1 + s['n_pi'] + len(s['fanins'])
        t = np.full(n, 2, dtype=np.int64); t[0] = 0; t[1:1 + s['n_pi']] = 1
        typ.append(t)
        base = N + 1 + s['n_pi']
        f = s['fanins']
        fa.append(N + (f[:, 0] >> 1)); fb.append(N + (f[:, 1] >> 1))
        ia.append(f[:, 0] & 1); ib.append(f[:, 1] & 1)
        gid.append(base + np.arange(len(f)))
        prob.append(s['prob']); lev.append(s['lev'])
        N += n
    cat = lambda x: torch.from_numpy(np.concatenate(x))
    b = dict(N=N, typ=cat(typ), fa=cat(fa), fb=cat(fb), ia=cat(ia).float(), ib=cat(ib).float(),
             gid=cat(gid), prob=cat(prob).float(), offs=offs)
    lev = np.concatenate(lev)
    b['lev'] = torch.from_numpy(lev.astype(np.int64))
    glev = lev[b['gid'].numpy()]
    order = np.argsort(glev, kind='stable')
    b['groups'] = []
    gl = glev[order]
    bounds = np.flatnonzero(np.diff(gl)) + 1
    for chunk in np.split(order, bounds):
        b['groups'].append(torch.from_numpy(chunk))
    return b


class AIGEncoder(nn.Module):
    def z(self, h):
        """Embedding used for equivalence retrieval / contrastive alignment."""
        return self.proj(h) if self.proj is not None else h

    def __init__(self, d=64, rounds=2, proj=False):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)) if proj else None
        self.d, self.rounds = d, rounds
        self.emb = nn.Embedding(3, d)
        self.phi = nn.Sequential(nn.Linear(d + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.psi = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.gru = nn.GRUCell(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, b):
        h0 = self.emb(b['typ'])
        h = h0
        prev = h0
        for r in range(self.rounds):
            for g in b['groups']:
                nodes = b['gid'][g]
                a, c = b['fa'][g], b['fb'][g]
                ma = self.phi(torch.cat([h[a], b['ia'][g, None]], 1))
                mc = self.phi(torch.cat([h[c], b['ib'][g, None]], 1))
                m = ma + mc + self.psi(ma * mc)
                new = self.gru(m, prev[nodes])
                h = h.index_copy(0, nodes, new)
            prev = h
        p = torch.sigmoid(self.head(h)).squeeze(-1)
        return h, p


class AIGEncoderBi(AIGEncoder):
    """DeepGate-like bidirectional variant: attention-weighted fanin aggregation in the forward pass, followed
    by a reverse (fanout) pass in decreasing level order, as in DeepGate. Each round = forward + reverse."""

    def __init__(self, d=64, rounds=2, proj=False):
        super().__init__(d, rounds, proj)
        self.att = nn.Linear(2 * d, 1)
        self.phi_r = nn.Sequential(nn.Linear(d + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.gru_r = nn.GRUCell(d, d)

    def forward(self, b):
        h = self.emb(b['typ'])
        prev = h
        gid, fa, fb, ia, ib = b['gid'], b['fa'], b['fb'], b['ia'], b['ib']
        for r in range(self.rounds):
            for g in b['groups']:
                nodes = gid[g]; a, c = fa[g], fb[g]
                ma = self.phi(torch.cat([h[a], ia[g, None]], 1)); mc = self.phi(torch.cat([h[c], ib[g, None]], 1))
                hv = prev[nodes]
                wa = self.att(torch.cat([hv, ma], 1)); wc = self.att(torch.cat([hv, mc], 1))
                w = torch.softmax(torch.cat([wa, wc], 1), 1)
                m = 2 * (w[:, :1] * ma + w[:, 1:] * mc) + self.psi(ma * mc)
                h = h.index_copy(0, nodes, self.gru(m, hv))
            # reverse pass: each node aggregates messages from its fanout gates, highest level first
            for g in reversed(b['groups']):
                nodes = gid[g]; a, c = fa[g], fb[g]
                msg_a = self.phi_r(torch.cat([h[nodes], ia[g, None]], 1)); msg_c = self.phi_r(torch.cat([h[nodes], ib[g, None]], 1))
                agg = torch.zeros_like(h).index_add(0, a, msg_a).index_add(0, c, msg_c)
                tgt = torch.unique(torch.cat([a, c]))
                h = h.index_copy(0, tgt, self.gru_r(agg[tgt], h[tgt]))
            prev = h
        p = torch.sigmoid(self.head(h)).squeeze(-1)
        return h, p


class AIGTransformer(AIGEncoder):
    def __init__(self, d=128, rounds=2, proj=False, layers=2, heads=4, max_lev=256):
        super().__init__(d, rounds, proj)
        self.lev_emb = nn.Embedding(max_lev, d)
        lay = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers)
        self.max_lev = max_lev

    def forward(self, b):
        h, _ = AIGEncoder.forward(self, b)
        offs = list(b['offs']) + [b['N']]
        L = max(offs[i + 1] - offs[i] for i in range(len(offs) - 1))
        B = len(offs) - 1
        idx = torch.full((B, L), -1, dtype=torch.long, device=h.device)
        for i in range(B):
            n = offs[i + 1] - offs[i]
            idx[i, :n] = torch.arange(offs[i], offs[i + 1], device=h.device)
        pad = idx < 0
        x = h[idx.clamp(min=0)] + self.lev_emb(b['lev'][idx.clamp(min=0)].clamp(max=self.max_lev - 1))
        y = self.tf(x, src_key_padding_mask=pad)
        h = h.index_copy(0, idx[~pad], y[~pad])
        p = torch.sigmoid(self.head(h)).squeeze(-1)
        return h, p


ARCHS = {'gru': AIGEncoder, 'bi': AIGEncoderBi, 'tf': AIGTransformer}


def build(arch='gru', d=64, rounds=2, proj=False):
    return ARCHS[arch](d, rounds, proj)


def to_device(b, dev):
    """Move a collated batch to a device (groups and index tensors included)."""
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=True)
        elif k == 'groups':
            out[k] = [g.to(dev, non_blocking=True) for g in v]
        else:
            out[k] = v
    return out
