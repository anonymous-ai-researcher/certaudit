"""Design-level comparison on ISCAS (17 designs, test-only, models trained on EPFL), same procedure as the
EPFL table: per-design mean over circuits and 3 seeds, mean difference, cluster bootstrap CI, two-sided sign
test, exact sign-flip permutation test, Wilcoxon; Holm over the regime comparisons of each metric."""
import glob, pickle, json, itertools, numpy as np
from scipy.stats import wilcoxon
from analysis import group_metrics, design_test, SYN, INJ
import os
REG = [('Baseline', 'base'), ('Augmentation', 'aug'), ('TT-sim', f"ttsim_ph_l{os.environ.get('TTLAM', '0.1')}"),
       ('Circuit contrast', f"circ_ph_l{os.environ.get('CIRCLAM', '0.2')}"), ('EqAlign', f"eqalign_ph_l{os.environ.get('LAMSTAR', '0.2')}")]
RAW = {n: [pickle.load(open(x, 'rb')) for x in sorted(glob.glob(f'rawis_{p}_s*.pkl'))] for n, p in REG}
print({n: len(v) for n, v in RAW.items()})
gm = lambda key: (lambda R: group_metrics(R, kinds=SYN + INJ)[key])
def perm_p(d):
    d = np.asarray(d); n = len(d); obs = abs(d.mean())
    if n <= 20:
        s = np.array(list(itertools.product([1, -1], repeat=n)))
        return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj.tolist()
OUT = {}
for key, sc, lower in [('top1_rs', 100, False), ('incons_rs', 1, True)]:
    rows = []
    for a in ['Baseline', 'Augmentation', 'TT-sim', 'Circuit contrast']:
        r = design_test(RAW[a], RAW['EqAlign'], gm(key), B=10000)
        d = np.array(r['b']) - np.array(r['a'])
        better = int((d < 0).sum()) if lower else int((d > 0).sum())
        try: pw = float(wilcoxon(d).pvalue)
        except ValueError: pw = float('nan')
        rows.append(dict(vs=a, diff=sc * r['diff'], lo=sc * r['lo'], hi=sc * r['hi'], better=better, n=r['n'], p_sign=r['p_sign'], p_perm=perm_p(d), p_wil=pw))
    for r_, h in zip(rows, holm([r_['p_perm'] for r_ in rows])):
        r_['p_perm_holm'] = h
    OUT[key] = rows
    for r_ in rows:
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r_.items()})
json.dump(OUT, open('iscas_design_test.json', 'w'), indent=1)
# macros for the paper (appended after make_numbers2 / make_external in build.sh)
t = OUT['top1_rs']
M = dict(IsDesTopMin=f"{min(r['diff'] for r in t):.1f}", IsDesTopMax=f"{max(r['diff'] for r in t):.1f}",
         IsDesN=str(t[0]['n']), IsDesBetterAll='yes' if all(r['better'] == r['n'] for r in t) else 'no',
         IsDesHolmP=f"{max(r['p_perm_holm'] for r in t):.1g}")
with open('paper/numbers.tex', 'a') as fh:
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(M, {r['vs']: r['p_perm_holm'] for r in t})
