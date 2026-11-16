"""Audit an external circuit model (e.g., DeepGate2/3/4) with exactly the metrics of the paper.

Fill in `load_model` and `infer` below, then
    python audit_external.py --pack . --out results_deepgate2.json

infer(model, aag_path) must return
    emb  : float array [n_vars, d]  node embeddings indexed by AIGER variable index
    prob : float array [n_vars]     predicted signal probability (or None if the model has no prob head)
Rows for variable 0 (const) and PIs may be anything; only AND-gate rows are scored.
If your model renumbers nodes internally (DeepGate parsers usually do), map back to AIGER variable order.

Metrics (all on SAT-proven equivalent gate pairs; see README):
  mae_orig, mae_m_orig, mae_m_var  probability error on originals and on matched gates of each variant
  incons, incons_rs, err_rs, rho   self-inconsistency (all pairs / restructured pairs) and its ratio to error
  top1, top1_rs, chance            retrieval of the proven equivalent (pessimistic ties) and chance level
  auroc, auroc_hard                cosine AUROC vs random / same-level negatives
  adv worst                        largest matched-gate degradation over the 16 adversarial variants"""
import argparse, json, numpy as np


def load_model():
    # ---- FILL IN: e.g. DeepGate2 ----
    raise NotImplementedError('load your pretrained model here')


def infer(model, aag_path):
    # ---- FILL IN ----
    raise NotImplementedError


def read_aag(path):
    with open(path) as f:
        hdr = f.readline().split()
        M, I, L, O, A = map(int, hdr[1:6])
        for _ in range(I):
            f.readline()
        for _ in range(O):
            f.readline()
        fan = [tuple(map(int, f.readline().split()[1:3])) for _ in range(A)]
    return I, fan


def shash(path):
    """Structural hash of every variable's fanin cone (PIs keyed by index, AND commutative)."""
    I, fan = read_aag(path)
    H = [hash(('c',))] + [hash(('pi', k)) for k in range(I)]
    for a, b in fan:
        x = (H[a >> 1], a & 1); y = (H[b >> 1], b & 1)
        H.append(hash(('and',) + tuple(sorted([x, y]))))
    return np.array(H, dtype=object), I, len(fan)


def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def auroc(pos, neg):
    v = np.concatenate([pos, neg]); lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    r = v.argsort().argsort() + 1
    return (r[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def levels(path):
    I, fan = read_aag(path)
    L = [0] * (1 + I)
    for a, b in fan:
        L.append(1 + max(L[a >> 1], L[b >> 1]))
    return np.array(L)


def pair_metrics(c, nm, model, ho, po, yo, sho, lev, go):
    hv, pv = infer(model, f'{c}/{nm}.aag')
    yv = np.load(f'{c}/{nm}_prob.npy')
    m = np.load(f'{c}/{nm}_match.npz'); I, J, P = m['I'], m['J'], m['P']
    out = {}
    if len(I) == 0:
        return out
    shv, _, _ = shash(f'{c}/{nm}.aag')
    rs = sho[I] != shv[J]
    if pv is not None:
        out['mae_m_var'] = float(np.abs(pv[J] - yv[J]).mean()); out['mae_m_orig'] = float(np.abs(po[I] - yo[I]).mean())
        pj = np.where(P == 1, 1 - pv[J], pv[J]); diff = np.abs(po[I] - pj)
        out['incons'] = float(diff.mean())
        if rs.any():
            out['incons_rs'] = float(diff[rs].mean()); out['err_rs'] = float(np.abs(po[I[rs]] - yo[I[rs]]).mean())
    keep = P == 0
    if keep.any():
        zo = norm(ho[go]); zj = norm(hv[J[keep]]); sims = zj @ zo.T
        true = sims[np.arange(len(zj)), I[keep] - go[0]]
        hit = (sims >= true[:, None] - 1e-7).sum(1) == 1
        out['top1'] = float(hit.mean()); out['chance'] = 1.0 / len(go)
        if rs[keep].any():
            out['top1_rs'] = float(hit[rs[keep]].mean())
        rng = np.random.default_rng(0)
        ridx = rng.integers(0, len(go), size=len(true)); nmk = go[ridx] != I[keep]
        if nmk.any():
            out['auroc'] = float(auroc(true, sims[np.arange(len(true)), ridx][nmk]))
        hp, hn = [], []
        for t, i in enumerate(I[keep]):
            same = np.flatnonzero((lev[go] == lev[i]) & (go != i))
            if len(same):
                hp.append(true[t]); hn.append(sims[t, same[np.random.default_rng(t).integers(len(same))]])
        if hp:
            out['auroc_hard'] = float(auroc(np.array(hp), np.array(hn)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default='.')
    ap.add_argument('--out', default='results_external.json')
    a = ap.parse_args()
    index = json.load(open(f'{a.pack}/index.json'))
    model = load_model()
    R = {}
    for e in index:
        c = f'{a.pack}/circuits/c{e["id"]:04d}'
        sho, I, A = shash(f'{c}/orig.aag'); go = np.arange(1 + I, 1 + I + A); lev = levels(f'{c}/orig.aag')
        ho, po = infer(model, f'{c}/orig.aag'); yo = np.load(f'{c}/orig_prob.npy')
        if po is not None:
            R.setdefault('orig', {}).setdefault('mae_orig', []).append(float(np.abs(po[go] - yo[go]).mean()))
        worst = None
        for v in e['variants']:
            r = pair_metrics(c, v['name'], model, ho, po, yo, sho, lev, go)
            fam = 'adv' if v['name'].startswith('adv') else v['name']
            for k, x in r.items():
                R.setdefault(fam, {}).setdefault(k, []).append(x)
            if fam == 'adv' and 'mae_m_var' in r:
                g = r['mae_m_var'] - r['mae_m_orig']
                if worst is None or g > worst[0]:
                    worst = (g, r['mae_m_var'], r['mae_m_orig'])
        if worst:
            R.setdefault('adv_worst', {}).setdefault('worst', []).append(worst[1])
            R['adv_worst'].setdefault('orig', []).append(worst[2])
    S = {f: {k: float(np.nanmean(v)) for k, v in d.items()} for f, d in R.items()}
    for f, d in S.items():
        if 'incons_rs' in d and 'err_rs' in d:
            d['rho'] = d['incons_rs'] / d['err_rs']
    json.dump(S, open(a.out, 'w'), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
