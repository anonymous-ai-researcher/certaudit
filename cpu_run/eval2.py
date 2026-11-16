"""Final evaluation: per-circuit raw metrics for every model tag (for CIs, tests and figures).
Usage: python eval2.py tag1 tag2 ...  (loads v_<tag>.pt, writes raw_<tag>.pkl)"""
import sys, os, pickle, numpy as np, torch
from model import AIGEncoder, collate
from evaluate import shash, gates

torch.set_num_threads(int(os.environ.get('THREADS', 2)))
# DATA_TAG=hyp evaluates the confirmatory design instead (data_hyp.pkl, heldout_hyp.pkl, no
# adversarial set) and writes <tag>_raw/raw_<model>_<tag>.pkl (outside the main globs), leaving every main-set result untouched.
_DT = os.environ.get('DATA_TAG')
D = pickle.load(open(f'data_{_DT}.pkl' if _DT else 'data.pkl', 'rb'))
A = [[] for _ in D['test']] if _DT else pickle.load(open('adv.pkl', 'rb'))
_HP = f'heldout_{_DT}.pkl' if _DT else 'heldout.pkl'
H = pickle.load(open(_HP, 'rb')) if os.path.exists(_HP) else None
test, tvars = D['test'], D['tvars']
KINDS = list(tvars[0].keys())


@torch.no_grad()
def run(m, s):
    h, p = m(collate([s])); return (h if os.environ.get('RET_H') else m.z(h)).numpy(), p.numpy()


def auroc(pos, neg):
    v = np.concatenate([pos, neg]); lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    r = v.argsort().argsort() + 1
    return (r[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def pair_metrics(d, s, I, J, P, ho, po, hv, pv, sho, n):
    out = {}
    go, gv = gates(d), gates(s)
    out['mae'] = float(np.abs(pv[gv] - s['prob'][gv]).mean())          # all variant gates (incl. new ones)
    out['frac_matched'] = float(len(np.unique(J)) / max(len(gv), 1))
    if len(I) == 0:
        return out
    # matched-gate errors: same functions in both circuits (variant label == orig label or its complement)
    out['mae_m_var'] = float(np.abs(pv[J] - s['prob'][J]).mean())
    out['mae_m_orig'] = float(np.abs(po[I] - d['prob'][I]).mean())
    pj = np.where(P == 1, 1 - pv[J], pv[J])
    diff = np.abs(po[I] - pj)
    rs = sho[I] != shash(s)[J]
    out['incons'] = float(diff.mean()); out['frac_rs'] = float(rs.mean()); out['n_pairs'] = int(len(I))
    pos_o = set((d['pos'] >> 1).tolist())
    mpo = np.array([i in pos_o for i in I])
    if mpo.any():
        out['incons_po'] = float(diff[mpo].mean())
    if rs.any():
        out['incons_rs'] = float(diff[rs].mean())
        out['err_rs'] = float(np.abs(po[I[rs]] - d['prob'][I[rs]]).mean())
        out['err_rs_var'] = float(np.abs(pv[J[rs]] - s['prob'][J[rs]]).mean())
    zo = ho[go] / (np.linalg.norm(ho[go], axis=1, keepdims=True) + 1e-9)
    keep = P == 0
    if keep.any():
        zj = hv[J[keep]] / (np.linalg.norm(hv[J[keep]], axis=1, keepdims=True) + 1e-9)
        sims = zj @ zo.T
        true = sims[np.arange(len(zj)), I[keep] - go[0]]
        # pessimistic tie-breaking: a hit only if no other original gate is at least as similar
        hit = ((sims >= true[:, None] - 1e-7).sum(1) == 1)
        out['top1'] = float(hit.mean())
        out['chance'] = float(1.0 / len(go))
        # hard negatives: original gates at the same logic level as the true equivalent
        lev = d['lev']; hn_pos, hn_neg = [], []
        for t_, i_ in enumerate(I[keep]):
            same = np.flatnonzero((lev[go] == lev[i_]) & (go != i_))
            if len(same):
                hn_pos.append(true[t_]); hn_neg.append(sims[t_, same[np.random.default_rng(t_).integers(len(same))]])
        if hn_pos:
            out['auroc_hard'] = float(auroc(np.array(hn_pos), np.array(hn_neg)))
        rk = rs[keep]
        if rk.any():
            out['top1_rs'] = float(hit[rk].mean())
        rng = np.random.default_rng(n)
        ps = (zj * zo[I[keep] - go[0]]).sum(1)
        ridx = rng.integers(0, len(go), size=len(ps)); nm = go[ridx] != I[keep]
        ns = sims[np.arange(len(ps)), ridx][nm]
        if len(ns):
            out['auroc'] = float(auroc(ps, ns))
    out['_pairs'] = dict(lev=d['lev'][I], rs=rs, diff=diff, err=np.abs(po[I] - d['prob'][I]))
    return out


def evaluate(m):
    R = []
    for n, d in enumerate(test):
        ho, po = run(m, d); go = gates(d); sho = shash(d)
        rec = dict(src=d['src'], mae_orig=float(np.abs(po[go] - d['prob'][go]).mean()), kinds={}, adv=[])
        for k in KINDS:
            v = tvars[n][k]; hv, pv = run(m, v['var'])
            rec['kinds'][k] = pair_metrics(d, v['var'], v['I'], v['J'], v['P'], ho, po, hv, pv, sho, n)
        for v in ([] if os.environ.get('NOADV') else A[n]):
            hv, pv = run(m, v['var'])
            r = pair_metrics(d, v['var'], v['I'], v['J'], v['P'], ho, po, hv, pv, sho, n)
            r.pop('_pairs', None); rec['adv'].append(r)
        if H is not None and H[n] is not None:
            rec['heldout'] = {}
            for k in ('H1', 'H2'):
                v = H[n][k]; hv, pv = run(m, v['var'])
                rec['heldout'][k] = pair_metrics(d, v['var'], v['I'], v['J'], v['P'], ho, po, hv, pv, sho, n)
        R.append(rec)
    return R


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        from model import build
        m = build(tag); m.load_state_dict(torch.load(f'v_{tag}.pt')); m.eval()
        pickle.dump(evaluate(m), open((f'{_DT}_raw/raw_{tag}_{_DT}' if _DT else (f'hret_raw/raw_{tag}_hret' if os.environ.get('RET_H') else f'raw_{tag}')) + '.pkl', 'wb'))
        print('evaluated', tag, flush=True)
