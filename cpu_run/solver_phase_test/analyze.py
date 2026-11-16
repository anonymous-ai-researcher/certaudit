"""Analysis of the solver phase test (rules fixed before any solve). Writes summary.json next to this file."""
import os, json, glob, math, itertools, numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, 'res', '*.jsonl'))) for l in open(f)]
REGS = ['Base', 'Aug', 'TT', 'Circ', 'Ours']
lc = lambda c: math.log2(c + 1)
C = {(r['design'], r['mut'], r['X'], r['src'], r['seed'], r['tau'], r['mode']): r for r in R}
inst = sorted({(r['design'], r['mut'], r['X']) for r in R})
hard = {i for i in inst if C.get(i + ('default', 0, 0, 'own'), {}).get('conf', 0) >= 1000}


def sign_p(x):
    x = [v for v in x if v != 0]; n = len(x)
    if n == 0:
        return float('nan'), 0, 0
    k = sum(v > 0 for v in x); m = min(k, n - k)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)
    return p, k, n


def summ(vals):
    v = np.array(vals, dtype=float)
    if len(v) == 0:
        return dict(n=0)
    p, k, n = sign_p(list(v))
    return dict(n=len(v), geo_ratio=float(2 ** v.mean()), mean_log2=float(v.mean()), median_log2=float(np.median(v)),
                n_pos=k, n_nonzero=n, sign_p=p, frac_2x=float(np.mean(np.abs(v) >= 1)))


def vs_default(src, tau, subset):
    out = []
    for i in subset:
        d = C.get(i + ('default', 0, 0, 'own'))
        if not d:
            continue
        seeds = [0] if src in ('oracle', 'random') else [0, 1, 2]
        x = [lc(C[i + (src, s, tau, 'own')]['conf']) - lc(d['conf']) for s in seeds if i + (src, s, tau, 'own') in C]
        if x:
            out.append(float(np.mean(x)))
    return summ(out)


def swap(reg, tau, mode, subset):
    out, ch = [], []
    for i in subset:
        if i[2] == 'D':
            continue
        x = [lc(C[i + (reg, s, tau, 'own')]['conf']) - lc(C[i + (reg, s, tau, mode)]['conf'])
             for s in (0, 1, 2) if i + (reg, s, tau, mode) in C and i + (reg, s, tau, 'own') in C]
        c = [C[i + (reg, s, tau, mode)]['changed'] for s in (0, 1, 2) if i + (reg, s, tau, mode) in C]
        if x:
            out.append(float(np.mean(x))); ch.append(float(np.mean(c)))
    s = summ(out); s['abs_median_log2'] = float(np.median(np.abs(out))) if out else float('nan')
    s['hints_changed_mean'] = float(np.mean(ch)) if ch else float('nan')
    return s


def spread(src, tau, subset):
    out = []
    for i in subset:
        if i[2] == 'D':
            continue
        seeds = [0] if src == 'oracle' else [0, 1, 2]
        x = [abs(lc(C[i + (src, s, tau, 'own')]['conf']) - lc(C[(i[0], i[1], 'D', src, s, tau, 'own')]['conf']))
             for s in seeds if i + (src, s, tau, 'own') in C and (i[0], i[1], 'D', src, s, tau, 'own') in C]
        if x:
            out.append(float(np.mean(x)))
    return dict(n=len(out), median_abs_log2=float(np.median(out)) if out else float('nan'),
                mean_abs_log2=float(np.mean(out)) if out else float('nan'))


S = dict(n_instances=len(inst), n_hard=len(hard), n_solves=len(R),
         budget_hits=sum(r['status'] == 'BUDGET' for r in R),
         per_design={d: sum(i[0] == d for i in inst) for d in sorted({i[0] for i in inst})})
S['default_status'] = {s: sum(C[i + ('default', 0, 0, 'own')]['status'] == s for i in inst) for s in ('SAT', 'UNSAT', 'BUDGET')}
for name, sub in (('all', inst), ('hard', sorted(hard))):
    T = S.setdefault(name, {})
    for tau in (0.01, 0.5):
        k = f'tau{tau}'
        T[k] = dict(P1={src: vs_default(src, tau, sub) for src in ['oracle', 'random'] + REGS},
                    P2={reg: swap(reg, tau, 'swap', sub) for reg in REGS},
                    placebo={reg: swap(reg, tau, 'placebo', sub) for reg in ('Base', 'Ours')},
                    P3_spread={src: spread(src, tau, sub) for src in ['oracle'] + REGS})
b = S['all']['tau0.01']['P2']['Base']; pl = S['all']['tau0.01']['placebo']['Base']
S['P2_rule'] = dict(geo_ratio=b.get('geo_ratio'), sign_p=b.get('sign_p'), abs_med=b.get('abs_median_log2'),
                    placebo_abs_med=pl.get('abs_median_log2'),
                    passes=bool(b.get('n', 0) and abs(b['geo_ratio'] - 1) >= 0.10 and b['sign_p'] < 0.05
                                and b['abs_median_log2'] > pl['abs_median_log2']))
json.dump(S, open(os.path.join(HERE, 'summary.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('wrote', os.path.join(HERE, 'summary.json'))
