"""Decision consequences of equivalence blindness (stage A, design fixed in advance:
design note). For every proven pair (v, u) the true probabilities are equal, so any
disagreement between the decisions taken on v and on u comes from structure alone.

  D1 SAT phase (CASCAD-style rule on unconditional probabilities): 0 if p < tau, 1 if p > 1 - tau, else solver
     default; tau in {0.005, 0.01, 0.05}.
  D2 testability: flag "hard to activate by random vectors" if p < 0.05 or p > 0.95; and per circuit and variant,
     the 5% of matched gates predicted hardest (smallest min(p, 1 - p)): Jaccard of the two candidate sets.

Reported per regime and family group (restructured pairs): flip rate, error rate of the decision on v against the
true decision, flip / error; design level (7 designs): mean over seeds of per-design rates, EqAlign and
augmentation against the baseline with an exact sign-flip test. Rule fixed in advance: structure matters for
downstream decisions if the baseline's phase flip rate under synthesis exceeds 25% of its error rate.
  python3 decisions.py [pairpred dir] [prefix map json]  ->  decisions.json"""
import os, sys, json, glob, itertools, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DIR = sys.argv[1] if len(sys.argv) > 1 else 'pairpred'
REG = json.loads(sys.argv[2]) if len(sys.argv) > 2 else \
    {'Base': 'base', 'Aug': 'aug', 'TT': 'ttsim_ph_l0.1', 'Circ': 'circ_ph_l0.2', 'Ours': 'eqalign_ph_l0.2'}
TAUS = [0.005, 0.01, 0.05]
GROUPS = {'syn': ['S1', 'S2', 'S3', 'S4'], 'inj': ["('inj', 0.1)", "('inj', 0.3)", "('inj', 0.5)"], 'H2': ['H2'], 'H1': ['H1']}


def phase(x, t):
    return np.where(x < t, 0, np.where(x > 1 - t, 1, -1))


def flag(x):
    return (x < 0.05) | (x > 0.95)


def jaccard_top(z, mask, frac=0.05):
    """Per (circuit, family) group: Jaccard of the top-frac hardest pairs by v's and by u's prediction."""
    out = []
    key = z['circ'][mask] * 100 + z['fam'][mask]
    pv, pu = z['pv'][mask], z['pu'][mask]; circ = z['circ'][mask]
    order = np.argsort(key, kind='stable'); key, pv, pu, circ = key[order], pv[order], pu[order], circ[order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(key)]):
        n = b - a; k = max(1, int(np.ceil(frac * n)))
        if n < 20:
            continue
        sv = set(np.argsort(np.minimum(pv[a:b], 1 - pv[a:b]), kind='stable')[:k])
        su = set(np.argsort(np.minimum(pu[a:b], 1 - pu[a:b]), kind='stable')[:k])
        out.append((circ[a], len(sv & su) / len(sv | su)))
    return out


def metrics(z, mask):
    p, pv, pu = z['p'][mask], z['pv'][mask], z['pu'][mask]
    r = {}
    for t in TAUS:
        dv, du, dt = phase(pv, t), phase(pu, t), phase(p, t)
        r[f'flip_{t}'] = float((dv != du).mean()); r[f'err_{t}'] = float((dv != dt).mean())
        r[f'opp_{t}'] = float(((dv >= 0) & (du >= 0) & (dv != du)).mean())          # 0 against 1: opposite phases
        r[f'truecommit_{t}'] = float((dt >= 0).mean())
        act = (dv >= 0) | (du >= 0)                       # a fixed phase from at least one implementation
        r[f'cflip_{t}'] = float((dv != du)[act].mean()) if act.any() else float('nan')
    fv, fu, ft = flag(pv), flag(pu), flag(p)
    r['tflip'] = float((fv != fu).mean()); r['terr'] = float((fv != ft).mean()); r['ttrue'] = float(ft.mean())
    ta = fv | fu
    r['ctflip'] = float((fv != fu)[ta].mean()) if ta.any() else float('nan')
    return r


src_names = None
res = {}
for reg, pre in REG.items():
    files = sorted(f for f in glob.glob(f'{DIR}/{pre}_s*.npz'))
    per_seed = []
    for f in files:
        z = dict(np.load(f, allow_pickle=False)); fams = list(z['fams']); src = z['src']
        src_names = sorted(set(src))
        des = src[z['circ']]
        S = {}
        for g, fl in GROUPS.items():
            fm = np.isin(z['fam'], [fams.index(x) for x in fl if x in fams]) & z['rs']
            S[g] = dict(pooled=metrics(z, fm), design={d: metrics(z, fm & (des == d)) for d in src_names})
            J = jaccard_top(z, fm)
            S[g]['jacc'] = float(np.mean([j for _, j in J]))
            S[g]['jacc_design'] = {d: (float(np.mean(x)) if (x := [j for c, j in J if src[c] == d]) else float('nan')) for d in src_names}
        per_seed.append(S)
    res[reg] = per_seed

# aggregate over seeds; design-level tests against the baseline
def mean_seed(reg, g, key, design=None):
    v = [(s[g]['design'][design] if design else s[g]['pooled'])[key] if key != 'jacc' else
         (s[g]['jacc_design'][design] if design else s[g]['jacc']) for s in res[reg]]
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0


def signflip(d):
    d = np.asarray(d); obs = abs(d.mean())
    allm = [abs((d * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean([m >= obs - 1e-12 for m in allm]))


KEYS = [f'flip_{t}' for t in TAUS] + [f'err_{t}' for t in TAUS] + [f'opp_{t}' for t in TAUS] + [f'cflip_{t}' for t in TAUS] + \
       ['tflip', 'terr', 'ctflip', 'jacc']
OUT = {'regimes': list(REG), 'designs': src_names, 'n_seeds': {r: len(v) for r, v in res.items()}, 'pooled': {}, 'tests': {}}
for g in GROUPS:
    OUT['pooled'][g] = {r: {k: mean_seed(r, g, k) for k in KEYS} for r in REG}
    OUT['pooled'][g]['_truecommit'] = {str(t): res['Base'][0][g]['pooled'][f'truecommit_{t}'] for t in TAUS}
    OUT['pooled'][g]['_ttrue'] = res['Base'][0][g]['pooled']['ttrue']
    OUT['tests'][g] = {}
    for r in REG:
        if r == 'Base':
            continue
        T = {}
        for k in KEYS:
            dd = [mean_seed(r, g, k, d)[0] - mean_seed('Base', g, k, d)[0] for d in src_names]
            dd = [x for x in dd if np.isfinite(x)]
            better = sum((x > 0) if k == 'jacc' else (x < 0) for x in dd)
            T[k] = dict(diff=float(np.mean(dd)), better=f'{better}/{len(dd)}', p=signflip(dd))
        OUT['tests'][g][r] = T
b = OUT['pooled']['syn']['Base']
OUT['rule'] = dict(flip=b['flip_0.01'][0], err=b['err_0.01'][0], ratio=b['flip_0.01'][0] / b['err_0.01'][0],
                   passes=b['flip_0.01'][0] > 0.25 * b['err_0.01'][0])
json.dump(OUT, open(os.environ.get('DEC_OUT', 'decisions.json'), 'w'), indent=1)

pct = lambda x: f'{100 * x:5.2f}'
for g in ('syn', 'inj', 'H2'):
    print(f'\n== {g} (restructured pairs); true decisions committed at tau=0.01: {pct(OUT["pooled"][g]["_truecommit"]["0.01"])}%, '
          f'true hard-to-activate: {pct(OUT["pooled"][g]["_ttrue"])}%')
    print('regime  | phase flip% t=.005 .01 .05 | phase err% .01 | flip|active% .01 | test-flag flip% err% flip|flagged% | top5% Jaccard')
    for r in REG:
        P = OUT['pooled'][g][r]
        print(f"{r:7s} | {pct(P['flip_0.005'][0])} {pct(P['flip_0.01'][0])} {pct(P['flip_0.05'][0])} | {pct(P['err_0.01'][0])} | "
              f"{pct(P['cflip_0.01'][0])} | {pct(P['tflip'][0])} {pct(P['terr'][0])} {pct(P['ctflip'][0])} | {P['jacc'][0]:.3f}")
    for r in ('Aug', 'Ours'):
        T = OUT['tests'][g][r]
        print(f"  {r} - Base: phase flip .01 {100*T['flip_0.01']['diff']:+.2f} ({T['flip_0.01']['better']}, p {T['flip_0.01']['p']:.3f}); "
              f"test flip {100*T['tflip']['diff']:+.2f} ({T['tflip']['better']}, p {T['tflip']['p']:.3f}); "
              f"Jaccard {T['jacc']['diff']:+.3f} ({T['jacc']['better']}, p {T['jacc']['p']:.3f}); flip|active {100*T['cflip_0.01']['diff']:+.1f} ({T['cflip_0.01']['better']})")
print('\nRule (baseline, synthesis, tau=0.01): flip', pct(OUT['rule']['flip']), '% vs error', pct(OUT['rule']['err']),
      '% -> ratio', round(OUT['rule']['ratio'], 2), '->', 'PASS (structure matters)' if OUT['rule']['passes'] else 'below threshold')
