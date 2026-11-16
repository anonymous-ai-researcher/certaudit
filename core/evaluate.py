"""Evaluation with exactly the paper's per-circuit metrics (pair_metrics is eval2.py unchanged).

  python -m core.evaluate --run runs/x [--suites val,epfl,iscas]

Writes into the run directory:
  val.json        validation metrics (used ONLY for selection: MAE, restructured Top-1, inconsistency ratio)
  raw_epfl.pkl    per-circuit records for the 376 EPFL test circuits: 7 fixed families, 16 adversarial
                  variants, held-out H1/H2 (same structure as the paper's raw_<tag>.pkl)
  raw_iscas.pkl   per-circuit records for the ISCAS85/89 suite (fixed + held-out, no adversarial set)
  raw_abc.pkl     (suite abc, E17) the EPFL test cones against their certified ABC rewrites (data/abc.pkl)
  raw_abc_iscas.pkl (suite abc_iscas, E17) the same for the ISCAS85/89 test cones
  raw_epfl_h.pkl  (suite epfl_h) EPFL again with retrieval on the encoder state h instead of the projection
All files are written atomically (temp file + rename).
"""
import argparse, json, os, pickle, numpy as np, torch
from core.model import build, collate, to_device

DATA_DIR = os.environ.get('CERT_DATA', 'data')
ABC_FILE = os.environ.get('ABC_FILE', 'abc.pkl')       # E17 smoke test: ABC_FILE=abc_smoke.pkl


def gates(s):
    return np.arange(1 + s['n_pi'], 1 + s['n_pi'] + len(s['fanins']))


def shash(s):
    """Structural hash of each node's fanin cone (PIs keyed by index, commutative AND)."""
    n = 1 + s['n_pi'] + len(s['fanins'])
    H = [hash(('c',))] + [hash(('pi', k)) for k in range(s['n_pi'])]
    for a, b in s['fanins']:
        x = (H[int(a) >> 1], int(a) & 1); y = (H[int(b) >> 1], int(b) & 1)
        H.append(hash(('and',) + tuple(sorted([x, y]))))
    return np.array(H, dtype=object)


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
        # independent-error reference (the paper's rho_null2.py): variant-side signed errors permuted within
        # the variant, 20 permutations; gap closed g = 1 - incons_rs / incons_perm
        y_ = d['prob'][I[rs]]; ev_ = po[I[rs]] - y_; eu_ = pj[rs] - y_
        rng_ = np.random.default_rng(1_000_003 * n + len(I))
        out['incons_perm'] = float(np.mean([np.abs(ev_ - rng_.permutation(eu_)).mean() for _ in range(20)]))
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



class Infer:
    """Our encoders: (z(h), p) for every node, z = projection head when the model has one."""

    def __init__(self, m, dev, use_h=False):
        self.m, self.dev, self.use_h = m, dev, use_h

    @torch.no_grad()
    def __call__(self, s):
        h, p = self.m(to_device(collate([s]), self.dev))
        z = h if self.use_h else self.m.z(h)
        return z.float().cpu().numpy(), p.float().cpu().numpy()


def eval_suite(run, test, tvars, adv=None, ho=None):
    R = []
    for n, d in enumerate(test):
        ho_, po = run(d); go = gates(d); sho = shash(d)
        rec = dict(src=d['src'], mae_orig=float(np.abs(po[go] - d['prob'][go]).mean()), kinds={}, adv=[])
        for k, v in tvars[n].items():
            hv, pv = run(v['var'])
            rec['kinds'][k] = pair_metrics(d, v['var'], v['I'], v['J'], v['P'], ho_, po, hv, pv, sho, n)
        for v in (adv[n] if adv is not None else []):
            hv, pv = run(v['var'])
            r = pair_metrics(d, v['var'], v['I'], v['J'], v['P'], ho_, po, hv, pv, sho, n)
            r.pop('_pairs', None); rec['adv'].append(r)
        if ho is not None and ho[n] is not None:
            rec['heldout'] = {}
            for k in ('H1', 'H2'):
                v = ho[n][k]; hv, pv = run(v['var'])
                rec['heldout'][k] = pair_metrics(d, v['var'], v['I'], v['J'], v['P'], ho_, po, hv, pv, sho, n)
        R.append(rec)
    return R


def val_metrics(run, D):
    """The paper's val_eval.py: 275 held-back training cones and their certified variant."""
    perm = np.random.default_rng(12345).permutation(len(D['train'])); VAL = np.sort(perm[:275])
    mae, top, inc, err = [], [], [], []
    for i in VAL:
        d, pr = D['train'][i], D['pairs'][i]
        s = pr['var']
        ho, po = run(d); hv, pv = run(s)
        go = gates(d)
        mae.append(np.abs(po[go] - d['prob'][go]).mean())
        I, J, P = pr['I'], pr['J'], pr['P']
        if len(I) == 0:
            continue
        rs = shash(d)[I] != shash(s)[J]
        pj = np.where(P == 1, 1 - pv[J], pv[J])
        if rs.any():
            inc.append(np.abs(po[I[rs]] - pj[rs]).mean()); err.append(np.abs(po[I[rs]] - d['prob'][I[rs]]).mean())
        keep = (P == 0) & rs
        if keep.any():
            zo = ho[go] / (np.linalg.norm(ho[go], axis=1, keepdims=True) + 1e-9)
            zj = hv[J[keep]] / (np.linalg.norm(hv[J[keep]], axis=1, keepdims=True) + 1e-9)
            top.append((go[(zj @ zo.T).argmax(1)] == I[keep]).mean())
    return dict(val_mae=float(np.mean(mae)), val_top1_rs=float(np.mean(top)),
                val_incons_rs_ratio=float(np.mean(inc) / np.mean(err)))


_CACHE = {}


def data(name):
    if name not in _CACHE:
        X = pickle.load(open(os.path.join(DATA_DIR, name), 'rb'))
        n = int(os.environ.get('CERT_SMOKE_N', 0))   # smoke test only: evaluate the first n test circuits
        if n:
            if name == 'data.pkl':
                X = dict(X, test=X['test'][:n], tvars=X['tvars'][:n])
            elif name in ('iscas.pkl', 'large.pkl'):
                X = dict(test=X['test'][:n], tvars=X['tvars'][:n], ho=X['ho'][:n])
            elif name == 'data_hyp.pkl':
                X = dict(X, test=X['test'][:n], tvars=X['tvars'][:n])
            elif name in ('adv.pkl', 'heldout.pkl', 'heldout_hyp.pkl'):
                X = X[:n]
            elif name in ('abc.pkl', 'abc_smoke.pkl'):
                X = dict(X, epfl=X['epfl'][:n], iscas=X['iscas'][:n])
        _CACHE[name] = X
    return _CACHE[name]


def _dump(obj, path, js=False):
    tmp = path + '.tmp'
    with open(tmp, 'w' if js else 'wb') as f:
        json.dump(obj, f, indent=1) if js else pickle.dump(obj, f)
    os.replace(tmp, path)


def evaluate_run(run_dir, infer, suites, infer_h=None):
    if 'val' in suites:
        _dump(val_metrics(infer, data('data.pkl')), os.path.join(run_dir, 'val.json'), js=True)
    if 'epfl' in suites:
        D = data('data.pkl')
        _dump(eval_suite(infer, D['test'], D['tvars'], data('adv.pkl'), data('heldout.pkl')), os.path.join(run_dir, 'raw_epfl.pkl'))
    if 'iscas' in suites:
        S = data('iscas.pkl')
        _dump(eval_suite(infer, S['test'], S['tvars'], None, S['ho']), os.path.join(run_dir, 'raw_iscas.pkl'))
    if 'hyp' in suites:                              # E14: confirmatory design hyp, evaluation only (nothing is tuned on it)
        Hd = data('data_hyp.pkl')
        _dump(eval_suite(infer, Hd['test'], Hd['tvars'], None, data('heldout_hyp.pkl')), os.path.join(run_dir, 'raw_hyp.pkl'))
    if 'abc' in suites or 'abc_iscas' in suites:     # E17: certified ABC rewrites of the same test cones (test only)
        A = data(ABC_FILE)
        if 'abc' in suites:
            D = data('data.pkl')
            _dump(eval_suite(infer, D['test'][:len(A['epfl'])], A['epfl']), os.path.join(run_dir, 'raw_abc.pkl'))
        if 'abc_iscas' in suites:
            S = data('iscas.pkl')
            _dump(eval_suite(infer, S['test'][:len(A['iscas'])], A['iscas']), os.path.join(run_dir, 'raw_abc_iscas.pkl'))
    if 'epfl_h' in suites and infer_h is not None:
        D = data('data.pkl')
        _dump(eval_suite(infer_h, D['test'], D['tvars'], None, data('heldout.pkl')), os.path.join(run_dir, 'raw_epfl_h.pkl'))


def load_run(run_dir, dev):
    cfg = json.load(open(os.path.join(run_dir, 'train.json')))['config']
    m = build(cfg['arch'], cfg['d'], cfg['rounds'], proj=cfg['method'].endswith('_ph'))
    m.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu'))
    return m.to(dev).eval(), cfg


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True); p.add_argument('--suites', default='val,epfl,iscas')
    p.add_argument('--device', default='auto')
    a = p.parse_args()
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    m, cfg = load_run(a.run, dev)
    evaluate_run(a.run, Infer(m, dev), a.suites.split(','), Infer(m, dev, use_h=True))
    print('evaluated', a.run, a.suites, flush=True)
