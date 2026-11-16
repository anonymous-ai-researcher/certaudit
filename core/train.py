"""Training, device-aware. The losses, batching, optimizer, schedule, clipping and the train/validation split are
the paper's train.py unchanged; only the architecture, width, depth, learning rate, number of certified variants
per training cone and the device are configurable.

  python -m core.train --method eqalign_ph --lam 0.2 --arch gru --d 64 --rounds 2 --seed 0 --out runs/x

Methods (as in the paper):
  base        signal-probability L1 on original circuits only
  aug         + the same loss on certified variants
  ttsim_ph    aug + DeepGate2-style truth-table-similarity loss (weight lam) through a projection head
  circ_ph     aug + FGNN-style circuit-level contrast (weight lam) through a projection head
  l1align     aug + MixGate-style L1 alignment of SAT-proven pairs on h (weight lam), positives only
  l1tt        l1align + TT-sim term (weight TTW, default 0.1)
  eqalign_ph  aug + EqAlign: InfoNCE on SAT-proven pairs (weight lam, tau 0.1) through a projection head
Ablations (the paper's appendix, now at every scale they are planned for):
  eqalign     EqAlign without the projection head (InfoNCE on h)
  cosalign_ph positives-only cosine alignment of SAT-proven pairs through the same head (no negatives)
  ttsimf_ph   DeepGate2-faithful TT-sim (DeepGate2's pair filters, zero-normalised distances)
  --probe-from RUN   freeze that run's encoder and train only the projection head with the method's loss
GPU note: CUDA scatter/index kernels are not bit-deterministic, so reruns on GPU differ slightly; on CPU the
code reproduces the paper's train.py loss trace exactly (checked for eqalign_ph and ttsim_ph).
"""
import argparse, json, os, pickle, time, numpy as np, torch, torch.nn.functional as F
from core.model import build, collate, to_device
from core.aiglib import Rep, simulate

DATA_DIR = os.environ.get('CERT_DATA', 'data')
METHODS = ('base', 'aug', 'ttsim_ph', 'circ_ph', 'l1align', 'l1tt', 'eqalign_ph', 'eqalign', 'cosalign_ph', 'ttsimf_ph')
_SIG = {}


def sig(s):
    """16-word (1,024-pattern) simulation signatures as {0,1} bit matrix, cached per sample."""
    k = id(s)
    if k not in _SIG:
        V = simulate(Rep(s['n_pi'], s['fanins'], s['pos']), 16, seed=7)
        _SIG[k] = np.unpackbits(V.view(np.uint8), axis=1)          # uint8 cache (4x less RAM), exact
    return _SIG[k].astype(np.float32)


def split(n):
    """The paper's fixed split of the training cones: 275 validation cones, the rest for training."""
    perm = np.random.default_rng(12345).permutation(n)
    return np.sort(perm[275:]), np.sort(perm[:275])


def load_pairs(D, V, extra_file='pairs_extra.pkl'):
    """List per training cone of its certified variants: the paper's variant plus V-1 extra ones."""
    pairs = [[p] for p in D['pairs']]
    if V > 1:
        X = pickle.load(open(os.path.join(DATA_DIR, extra_file), 'rb'))   # tools/gen_extra_variants.py
        assert len(X) == len(pairs) or extra_file.endswith('_smoke.pkl'), \
            f'{extra_file} has {len(X)} cones, data.pkl has {len(pairs)}: incomplete file, regenerate it'
        for i, ex in enumerate(X):
            pairs[i] += ex[:V - 1]
        short = sum(len(p) < V for p in pairs)
        print(f'variants per cone: {V} (cones with fewer certified variants available: {short})', flush=True)
    return pairs


HOLDOUT = ('S1', 'S2', 'S3', 'S4', 'inj')


def in_group(kind, holdout):
    return str(kind).startswith("('inj'") if holdout == 'inj' else str(kind) == holdout


def apply_holdout(pairs, holdout, extra_file):
    """E12 leave-one-family-out: remove every training variant of the held-out family (S1..S4, or both injection
    rates). A cone whose only variant is held out takes its first certified extra variant of another family
    (tools/gen_extra_variants.py, the same file E4 uses), so the number of training pairs stays the same; a cone
    with no such variant is dropped from training. Validation and test data are untouched."""
    X = pickle.load(open(os.path.join(DATA_DIR, extra_file), 'rb'))
    assert len(X) == len(pairs) or extra_file.endswith('_smoke.pkl'), f'{extra_file} incomplete: regenerate it'
    out, replaced, dropped = [], 0, 0
    for i, ps in enumerate(pairs):
        keep = [q for q in ps if not in_group(q['kind'], holdout)]
        if not keep:
            alt = [q for q in (X[i] if i < len(X) else []) if not in_group(q['kind'], holdout)]
            if alt:
                keep = alt[:1]; replaced += 1
            else:
                dropped += 1
        out.append(keep)
    print(f'holdout {holdout}: {replaced} cones re-paired with another family, {dropped} dropped', flush=True)
    return out, replaced, dropped


def main(a):
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    D = pickle.load(open(os.path.join(DATA_DIR, 'data.pkl'), 'rb'))
    train, pairs = D['train'], load_pairs(D, a.variants, a.extra_file)
    if a.pairs_file:
        # E14: the positive pairs of each training variant come from simulation signatures alone (data/pairs_sig64.pkl,
        # pairs_sig4096.pkl: first signature match accepted without a SAT proof, index-aligned with D['pairs']); the
        # variants, loss and everything else are unchanged, so training differs only in where the positives come from
        assert a.variants == 1, '--pairs-file replaces the pairs of the paper variant only'
        alt = pickle.load(open(os.path.join(DATA_DIR, a.pairs_file), 'rb'))
        assert len(alt) == len(pairs), f'{a.pairs_file}: {len(alt)} cones, data.pkl has {len(pairs)}'
        pairs = [[dict(ps[0], I=q['I'], J=q['J'], P=q['P'])] for ps, q in zip(pairs, alt)]
        print(f'positives from {a.pairs_file}: {sum(len(q["I"]) for q in alt)} pairs', flush=True)
    ho = (0, 0)
    if a.holdout:
        assert a.holdout in HOLDOUT, a.holdout
        pairs, *ho = apply_holdout(pairs, a.holdout, a.holdout_extra)
    TR, _ = split(len(train))
    TR = [i for i in TR if pairs[i]]                      # only cones dropped by --holdout can have no pair
    train = [train[i] for i in TR]; pairs = [pairs[i] for i in TR]
    method = a.method
    assert method in METHODS, method
    model = build(a.arch, a.d, a.rounds, proj=method.endswith('_ph')).to(dev)
    if a.probe_from:
        src = torch.load(os.path.join(a.probe_from, 'model.pt'), map_location='cpu')
        miss, unexp = model.load_state_dict(src, strict=False)
        assert all(k.startswith('proj.') for k in miss) and not unexp, (miss, unexp)
        for n_, p_ in model.named_parameters():
            p_.requires_grad = n_.startswith('proj.')
    opt = torch.optim.Adam([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr)
    BS, n = 16, len(train)
    steps_total = a.epochs * ((n + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps_total, pct_start=0.1,
                                                anneal_strategy='cos')
    vrng = np.random.default_rng(1000 + a.seed)          # which certified variant a cone uses in a step
    t0 = time.time(); log = []
    for ep in range(a.epochs):
        perm = np.random.permutation(n)
        tot, totc, nb = 0.0, 0.0, 0
        for k in range(0, n, BS):
            idx = perm[k:k + BS]
            if method == 'base':
                b = to_device(collate([train[i] for i in idx]), dev)
                h, p = model(b)
                loss = F.l1_loss(p[b['gid']], b['prob'][b['gid']])
                lc = torch.zeros((), device=dev)
            else:
                pk = [pairs[i][0] if len(pairs[i]) == 1 else pairs[i][int(vrng.integers(len(pairs[i])))] for i in idx]
                samples = [train[i] for i in idx] + [q['var'] for q in pk]
                bc = collate(samples); b = to_device(bc, dev)
                h, p = model(b)
                loss = F.l1_loss(p[b['gid']], b['prob'][b['gid']])
                lc = torch.zeros((), device=dev)
                if method in ('eqalign_ph', 'eqalign', 'cosalign_ph', 'l1align', 'l1tt'):
                    I, J = [], []
                    for q, pr in enumerate(pk):
                        keep = pr['P'] == 0
                        oa, ob = bc['offs'][q], bc['offs'][len(idx) + q]
                        I.append(oa + pr['I'][keep]); J.append(ob + pr['J'][keep])
                    I = torch.from_numpy(np.concatenate(I)); J = torch.from_numpy(np.concatenate(J))
                    if len(I) > 512:
                        sel = torch.randperm(len(I))[:512]; I, J = I[sel], J[sel]
                    I, J = I.to(dev), J.to(dev)
                    if method in ('eqalign_ph', 'eqalign'):
                        za = F.normalize(model.z(h[I]), dim=-1); zb = F.normalize(model.z(h[J]), dim=-1)
                        logits = za @ zb.t() / 0.1
                        tgt = torch.arange(len(I), device=dev)
                        lc = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
                    elif method == 'cosalign_ph':
                        lc = (1 - F.cosine_similarity(model.z(h[I]), model.z(h[J]), dim=-1)).mean()
                    else:
                        lc = (model.z(h[I]) - model.z(h[J])).abs().sum(-1).mean()
                    loss = loss + a.lam * lc
                if method == 'circ_ph':
                    nb_ = len(idx); Zs = []
                    gid_ = bc['gid'].numpy(); bounds = bc['offs'] + [bc['N']]
                    for q in range(2 * nb_):
                        lo, hi = bounds[q], bounds[q + 1]
                        sel = torch.from_numpy(gid_[(gid_ >= lo) & (gid_ < hi)]).to(dev)
                        Zs.append(model.z(h[sel]).mean(0))
                    Zs = F.normalize(torch.stack(Zs), dim=-1)
                    za, zb = Zs[:nb_], Zs[nb_:]
                    logits = za @ zb.t() / 0.1; tgt = torch.arange(nb_, device=dev)
                    lc = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
                    loss = loss + a.lam * lc
                if method == 'ttsimf_ph':
                    # the paper's DeepGate2-faithful TT-sim (train.py 'ttsimf'), unchanged
                    A_, B_, T_ = [], [], []
                    for q, s_ in enumerate(samples):
                        ng = len(s_['fanins'])
                        if ng < 2:
                            continue
                        base_ = 1 + s_['n_pi']; S_ = sig(s_)
                        lev_ = np.asarray(s_['lev'])[base_:base_ + ng]; p_ = S_[base_:base_ + ng].mean(1)
                        a_ = np.random.randint(0, ng, 2048); c_ = np.random.randint(0, ng, 2048)
                        dt_ = np.abs(S_[a_ + base_] - S_[c_ + base_]).mean(1)
                        ok = ((a_ != c_) & (np.abs(p_[a_] - p_[c_]) < 0.05) & (np.abs(lev_[a_] - lev_[c_]) <= 5)
                              & ((dt_ <= 0.2) | (dt_ >= 0.8)))
                        if not ok.any():
                            continue
                        a_, c_, dt_ = a_[ok][:64], c_[ok][:64], dt_[ok][:64]
                        T_.append(dt_); A_.append(bc['offs'][q] + a_ + base_); B_.append(bc['offs'][q] + c_ + base_)
                    if A_:
                        A_ = torch.from_numpy(np.concatenate(A_)).to(dev); B_ = torch.from_numpy(np.concatenate(B_)).to(dev)
                        DT = torch.from_numpy(np.concatenate(T_)).float().to(dev)
                        DH = 1 - F.cosine_similarity(model.z(h[A_]), model.z(h[B_]), dim=-1)
                        zn = lambda x: (x - x.mean()) / (x.std() + 1e-6)
                        lc = F.l1_loss(zn(DH), zn(DT)); loss = loss + a.lam * lc
                if method in ('ttsim_ph', 'l1tt'):
                    A_, B_, T_ = [], [], []
                    for q, s_ in enumerate(samples):
                        ng = len(s_['fanins'])
                        if ng < 2:
                            continue
                        base_ = 1 + s_['n_pi']
                        a_ = np.random.randint(0, ng, 64) + base_; c_ = np.random.randint(0, ng, 64) + base_
                        S_ = sig(s_)
                        T_.append(1 - np.abs(S_[a_] - S_[c_]).mean(1))
                        A_.append(bc['offs'][q] + a_); B_.append(bc['offs'][q] + c_)
                    A_ = torch.from_numpy(np.concatenate(A_)).to(dev); B_ = torch.from_numpy(np.concatenate(B_)).to(dev)
                    T_ = torch.from_numpy(np.concatenate(T_)).float().to(dev)
                    cs = F.cosine_similarity(model.z(h[A_]), model.z(h[B_]), dim=-1)
                    ltt = F.l1_loss(cs, T_)
                    if method == 'l1tt':
                        loss = loss + a.ttw * ltt
                    else:
                        lc = ltt; loss = loss + a.lam * lc
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss); totc += float(lc); nb += 1
        log.append(dict(epoch=ep, loss=tot / nb, lc=totc / nb, sec=time.time() - t0))
        print(f'{a.out} ep{ep} loss {tot / nb:.4f} lc {totc / nb:.3f} {time.time() - t0:.0f}s', flush=True)
        if not np.isfinite(tot):
            print('NON-FINITE LOSS: run marked as diverged', flush=True); break
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(vars(a)); cfg['n_params'] = sum(p.numel() for p in model.parameters())
    if a.holdout:
        import hashlib
        cfg['holdout_replaced'], cfg['holdout_dropped'] = ho
        cfg['holdout_extra_md5'] = hashlib.md5(open(os.path.join(DATA_DIR, a.holdout_extra), 'rb').read()).hexdigest()
    if a.variants > 1:
        import hashlib
        cfg['extra_file_md5'] = hashlib.md5(open(os.path.join(DATA_DIR, a.extra_file), 'rb').read()).hexdigest()
    if a.pairs_file:
        import hashlib
        cfg['pairs_file_md5'] = hashlib.md5(open(os.path.join(DATA_DIR, a.pairs_file), 'rb').read()).hexdigest()
    cfg['diverged'] = not np.isfinite(log[-1]['loss']); cfg['train_sec'] = time.time() - t0
    cfg['device_name'] = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'
    # atomic: model first (temp + rename), then train.json; a run counts as trained only when both exist
    torch.save(model.state_dict(), os.path.join(a.out, 'model.pt.tmp'))
    os.replace(os.path.join(a.out, 'model.pt.tmp'), os.path.join(a.out, 'model.pt'))
    json.dump(dict(config=cfg, log=log), open(os.path.join(a.out, 'train.json.tmp'), 'w'), indent=1)
    os.replace(os.path.join(a.out, 'train.json.tmp'), os.path.join(a.out, 'train.json'))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True); p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--arch', default='gru'); p.add_argument('--d', type=int, default=64)
    p.add_argument('--rounds', type=int, default=2); p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--epochs', type=int, default=12); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--variants', type=int, default=1); p.add_argument('--extra-file', default='pairs_extra.pkl'); p.add_argument('--ttw', type=float, default=0.1)
    p.add_argument('--device', default='auto'); p.add_argument('--out', required=True)
    p.add_argument('--probe-from', default=None)
    p.add_argument('--holdout', default=None, help='E12: hold out one training family (S1|S2|S3|S4|inj)')
    p.add_argument('--holdout-extra', default='pairs_extra.pkl')
    p.add_argument('--pairs-file', default=None, help='E14: training positives from simulation signatures instead of SAT')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
