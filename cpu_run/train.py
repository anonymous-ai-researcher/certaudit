"""Train three models: base (originals only), aug (originals + certified variants),
eqalign (aug + cross-variant functional contrastive alignment of embeddings)."""
import sys, os, time, pickle, numpy as np, torch, torch.nn.functional as F
from model import AIGEncoder, collate
from aiglib import Rep, simulate

_SIG = {}
def sig(s):
    """16-word (1,024-pattern) simulation signatures as {0,1} bit matrix, cached per sample."""
    k = id(s)
    if k not in _SIG:
        V = simulate(Rep(s['n_pi'], s['fanins'], s['pos']), 16, seed=7)
        _SIG[k] = np.unpackbits(V.view(np.uint8), axis=1).astype(np.float32)
    return _SIG[k]

import os; torch.set_num_threads(int(os.environ.get("THREADS", 2)))
method = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
torch.manual_seed(seed); np.random.seed(seed)
D = pickle.load(open('data.pkl', 'rb'))
train, pairs = D['train'], D['pairs']
# Experiment A: replace the SAT-proven positives by simulation-labelled ones (same variants).
if os.environ.get('PAIRS'):
    _alt = pickle.load(open(os.environ['PAIRS'], 'rb'))
    pairs = [dict(p, I=a['I'], J=a['J'], P=a['P']) for p, a in zip(pairs, _alt)]
# fixed train/validation split (validation used only for hyperparameter selection)
_perm = np.random.default_rng(12345).permutation(len(train))
VAL_IDX = np.sort(_perm[:275]); TR_IDX = np.sort(_perm[275:])
if os.environ.get('FULLTRAIN') is None:
    train = [train[i] for i in TR_IDX]; pairs = [pairs[i] for i in TR_IDX]
from model import AIGEncoderBi
Enc = AIGEncoderBi if method.startswith('bi_') else AIGEncoder
if method.startswith('bi_'):
    method_core = method[3:]
else:
    method_core = method
model = Enc(d=64, rounds=2, proj=method.endswith('_ph'))
# Experiment B: start from a trained encoder, freeze it, and train only the projection head with
# the EqAlign loss. Batches, pair sampling, temperature and loss are exactly EqAlign's.
if os.environ.get('PROBE_FROM'):
    _src = torch.load(os.environ['PROBE_FROM'])
    _miss, _unexp = model.load_state_dict(_src, strict=False)
    assert all(k.startswith('proj.') for k in _miss) and not _unexp, (_miss, _unexp)
    for _n, _p in model.named_parameters():
        _p.requires_grad = _n.startswith('proj.')
    print('probe: encoder frozen from', os.environ['PROBE_FROM'], '; trainable:',
          sum(p.numel() for p in model.parameters() if p.requires_grad), 'params', flush=True)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
steps_total = epochs * ((len(train) + 15) // 16)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps_total, pct_start=0.1, anneal_strategy='cos')
BS = 16
n = len(train)
t0 = time.time()
for ep in range(epochs):
    perm = np.random.permutation(n)
    tot, totc, nb = 0.0, 0.0, 0
    for k in range(0, n, BS):
        idx = perm[k:k + BS]
        if method_core == 'base':
            b = collate([train[i] for i in idx])
            h, p = model(b)
            loss = F.l1_loss(p[b['gid']], b['prob'][b['gid']])
            lc = torch.tensor(0.)
        else:
            samples = [train[i] for i in idx] + [pairs[i]['var'] for i in idx]
            b = collate(samples)
            h, p = model(b)
            loss = F.l1_loss(p[b['gid']], b['prob'][b['gid']])
            lc = torch.tensor(0.)
            if method_core.startswith('eqalign'):
                I, J = [], []
                for q, i in enumerate(idx):
                    pr = pairs[i]
                    keep = pr['P'] == 0
                    oa, ob = b['offs'][q], b['offs'][len(idx) + q]
                    I.append(oa + pr['I'][keep]); J.append(ob + pr['J'][keep])
                I = torch.from_numpy(np.concatenate(I)); J = torch.from_numpy(np.concatenate(J))
                if len(I) > 512:
                    sel = torch.randperm(len(I))[:512]; I, J = I[sel], J[sel]
                za = F.normalize(model.z(h[I]), dim=-1); zb = F.normalize(model.z(h[J]), dim=-1)
                logits = za @ zb.t() / 0.1
                tgt = torch.arange(len(I))
                lc = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
                loss = loss + lam * lc
            if method_core.startswith(('l1align', 'l1tt')):
                # MixGate-style alignment (Shi et al., arXiv 2509.20968, Eq. 4): mean L1 distance between the
                # functional embeddings of SAT-proven pairs, positives only, no negatives, no projection head.
                I, J = [], []
                for q, i in enumerate(idx):
                    pr = pairs[i]
                    keep = pr['P'] == 0
                    oa, ob = b['offs'][q], b['offs'][len(idx) + q]
                    I.append(oa + pr['I'][keep]); J.append(ob + pr['J'][keep])
                I = torch.from_numpy(np.concatenate(I)); J = torch.from_numpy(np.concatenate(J))
                if len(I) > 512:
                    sel = torch.randperm(len(I))[:512]; I, J = I[sel], J[sel]
                lc = (model.z(h[I]) - model.z(h[J])).abs().sum(-1).mean()
                loss = loss + lam * lc
            if method_core.startswith('circ'):
                # FGNN-style circuit-level contrastive baseline: mean-pooled projected embeddings of a circuit and
                # its certified variant are positives; other circuits in the batch are negatives. Same data as
                # EqAlign, but no gate-level correspondence.
                nb_ = len(idx); Zs = []
                gid_ = b['gid'].numpy()
                bounds = b['offs'] + [b['N']]
                for q in range(2 * nb_):
                    lo, hi = bounds[q], bounds[q + 1]
                    sel = torch.from_numpy(gid_[(gid_ >= lo) & (gid_ < hi)])
                    Zs.append(model.z(h[sel]).mean(0))
                Zs = F.normalize(torch.stack(Zs), dim=-1)
                za, zb = Zs[:nb_], Zs[nb_:]
                logits = za @ zb.t() / 0.1; tgt = torch.arange(nb_)
                lc = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
                loss = loss + lam * lc
            if method_core.startswith('ttsimf'):
                # DeepGate2-faithful variant: pairs are filtered as in DeepGate2 (same PIs by construction
                # within a cone; |p_i-p_j|<0.05; |lev_i-lev_j|<=5; truth-table distance <=0.2 or >=0.8) and
                # both distances are zero-normalised before the L1 loss.
                A_, B_, T_ = [], [], []
                for q, s_ in enumerate(samples):
                    ng = len(s_['fanins'])
                    if ng < 2:
                        continue
                    base_ = 1 + s_['n_pi']
                    S_ = sig(s_)
                    lev_ = np.asarray(s_['lev'])[base_:base_ + ng]
                    p_ = S_[base_:base_ + ng].mean(1)
                    a_ = np.random.randint(0, ng, 2048); c_ = np.random.randint(0, ng, 2048)
                    dt_ = np.abs(S_[a_ + base_] - S_[c_ + base_]).mean(1)
                    ok = ((a_ != c_) & (np.abs(p_[a_] - p_[c_]) < 0.05) & (np.abs(lev_[a_] - lev_[c_]) <= 5)
                          & ((dt_ <= 0.2) | (dt_ >= 0.8)))
                    if not ok.any():
                        continue
                    a_, c_, dt_ = a_[ok][:64], c_[ok][:64], dt_[ok][:64]
                    T_.append(dt_); A_.append(b['offs'][q] + a_ + base_); B_.append(b['offs'][q] + c_ + base_)
                if A_:
                    A_ = torch.from_numpy(np.concatenate(A_)); B_ = torch.from_numpy(np.concatenate(B_))
                    DT = torch.from_numpy(np.concatenate(T_)).float()
                    DH = 1 - F.cosine_similarity(model.z(h[A_]), model.z(h[B_]), dim=-1)
                    zn = lambda x: (x - x.mean()) / (x.std() + 1e-6)
                    lc = F.l1_loss(zn(DH), zn(DT))
                    loss = loss + lam * lc
            elif method_core.startswith(('ttsim', 'l1tt')):
                # DeepGate2-style functional-similarity loss: cosine of projected embeddings regresses the
                # truth-table similarity (1 - normalized Hamming distance) of random gate pairs per circuit.
                A_, B_, T_ = [], [], []
                for q, s_ in enumerate(samples):
                    ng = len(s_['fanins'])
                    if ng < 2:
                        continue
                    base_ = 1 + s_['n_pi']
                    a_ = np.random.randint(0, ng, 64) + base_; c_ = np.random.randint(0, ng, 64) + base_
                    S_ = sig(s_)
                    T_.append(1 - np.abs(S_[a_] - S_[c_]).mean(1)); A_.append(b['offs'][q] + a_); B_.append(b['offs'][q] + c_)
                A_ = torch.from_numpy(np.concatenate(A_)); B_ = torch.from_numpy(np.concatenate(B_))
                T_ = torch.from_numpy(np.concatenate(T_)).float()
                cs = F.cosine_similarity(model.z(h[A_]), model.z(h[B_]), dim=-1)
                lc = F.l1_loss(cs, T_)
                loss = loss + (float(os.environ.get('TTW', 0.1)) if method_core.startswith('l1tt') else lam) * lc
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        tot += float(loss); totc += float(lc); nb += 1
    print(f'{method} s{seed} ep{ep} loss {tot/nb:.4f} lc {totc/nb:.3f} {time.time()-t0:.0f}s', flush=True)
tag = method + (f'_l{lam:g}' if method_core.startswith(('eqalign', 'ttsim', 'ttsimf', 'circ', 'l1align', 'l1tt')) else '') + f'_s{seed}'
torch.save(model.state_dict(), f'v_{tag}.pt')
