"""Independent-error reference for the consistency ratio, computed inside eval2's own code path.

rho      = mean|p_v - p'_u| / mean|p_v - y_v|           on restructured proven pairs (Table I)
rho_perp = same numerator with the variant-side signed errors permuted within each variant,
           i.e. a model with the same accuracy on both sides whose two errors are independent
gap      = 1 - rho / rho_perp = 1 - mean|e_v - e_u| / mean|e_v - e_u_perm|
The denominator cancels in gap, so it cannot be lowered by being less accurate.

eval2.pair_metrics is wrapped (not copied) so rho is computed by the evaluator itself; the script
asserts that it reproduces the stored raw_<tag>.pkl value before reporting anything else.
Writes rho_null2.json with per-seed, per-family and per-design sums for design-level tests."""
import os, sys, json
os.environ['NOADV'] = '1'
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, torch
import eval2
from model import build
from analysis import load, group_summary, SYN, INJ
eval2.H = None                                     # fixed families only
_pm = eval2.pair_metrics


def pm(d, s, I, J, P, ho, po, hv, pv, sho, n):
    out = _pm(d, s, I, J, P, ho, po, hv, pv, sho, n)
    if len(I):
        rs = sho[I] != eval2.shash(s)[J]
        if rs.any():
            y = d['prob'][I[rs]]
            pj = np.where(P[rs] == 1, 1 - pv[J[rs]], pv[J[rs]])
            out['_ev'], out['_eu'] = po[I[rs]] - y, pj - y
    return out


eval2.pair_metrics = pm
NPERM = 20
OUT = json.load(open('rho_null2.json')) if os.path.exists('rho_null2.json') else {}
for tag in sys.argv[1:]:
    if tag in OUT:
        continue
    m = build(tag); m.load_state_dict(torch.load(f'v_{tag}.pt')); m.eval()
    R = eval2.evaluate(m)
    stored = group_summary(load(tag), kinds=SYN)[0]['rho']
    mine = group_summary(R, kinds=SYN)[0]['rho']
    assert abs(stored - mine) < 1e-6, f'{tag}: rho {mine} does not reproduce stored {stored}'
    rng = np.random.default_rng(0)
    res = {}
    for fam, kinds in (('syn', SYN), ('inj', INJ)):
        per_design = {}
        inc, perm, err = [], [], []
        for rec in R:
            for k in kinds:
                v = rec['kinds'][k]
                if '_ev' not in v:
                    continue
                ev, eu = v['_ev'], v['_eu']
                ip = float(np.mean([np.abs(ev - rng.permutation(eu)).mean() for _ in range(NPERM)]))
                inc.append(v['incons_rs']); perm.append(ip); err.append(v['err_rs'])
                pd = per_design.setdefault(rec['src'], [0.0, 0.0, 0])
                pd[0] += v['incons_rs']; pd[1] += ip; pd[2] += 1
        rho, rho_p = np.mean(inc) / np.mean(err), np.mean(perm) / np.mean(err)
        res[fam] = dict(rho=float(rho), rho_perp=float(rho_p), gap=float(1 - np.mean(inc) / np.mean(perm)),
                        design_gap={s_: 1 - a / b for s_, (a, b, c) in per_design.items()})
    OUT[tag] = res
    json.dump(OUT, open('rho_null2.json', 'w'), indent=1)
    print(f"{tag:32s} syn rho {res['syn']['rho']:.3f} (stored {stored:.3f}) perp {res['syn']['rho_perp']:.2f} "
          f"gap {100 * res['syn']['gap']:.0f}% | inj gap {100 * res['inj']['gap']:.0f}%", flush=True)
