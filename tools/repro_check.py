"""E0: does the GPU environment reproduce the paper's CPU numbers? Compares the 3-seed means of runs/E0 with
configs/paper_reference.json. A metric passes if |GPU - paper| <= max(abs tolerance, 3 x combined s.e.).
GPU arithmetic is not bit-identical to the CPU, so small differences are expected; a FAIL means stop and report.
  python -m tools.repro_check      -> results/E0/repro.md"""
import glob, json, os, pickle, numpy as np
from core.stats import seed_values, mean_sd_ci

TOL = {'MAE': 0.0015, 'rho_syn': 0.05, 'rho_inj': 0.06, 'rho_H2': 0.6, 'dMAE_H2': 0.008, 'incons_rs': 0.0012,
       'top1_seen': 1.5, 'top1_ho': 1.5, 'auroc_lvl': 0.01, 'worst_ratio': 0.06}
ref = json.load(open('configs/paper_reference.json'))['ref']
L = ['# E0 reproduction check (GPU vs the paper\'s CPU runs, 3 seeds each)\n',
     '| regime | metric | paper | GPU | abs diff | allowed | result |', '|---|---|---|---|---|---|---|']
nfail = 0
for m, R0 in ref.items():
    runs = sorted(glob.glob(f'runs/E0/gru_d64_r2/{m}_*_s[0-9]'))
    R = [pickle.load(open(os.path.join(r, 'raw_epfl.pkl'), 'rb')) for r in runs if os.path.exists(os.path.join(r, 'raw_epfl.pkl'))]
    if not R:
        L.append(f'| {m} | (no finished runs) | | | | | MISSING |'); nfail += 1; continue
    for k, tol in TOL.items():
        g = mean_sd_ci(seed_values(R, k)); p = R0[k]
        if p['mean'] != p['mean']:                 # not evaluated in the paper (e.g. no adversarial set): skip
            L.append(f"| {m} | {k} | n/a | {g['mean']:.4g} (n={g['n']}) | | | n/a |"); continue
        se = np.sqrt(g['sd'] ** 2 / max(g['n'], 1) + p['sd'] ** 2 / max(p['n'], 1))
        allow = max(tol, 3 * se); d = abs(g['mean'] - p['mean'])
        ok = d <= allow or (np.isnan(g['mean']) and np.isnan(p['mean']))
        nfail += not ok
        L.append(f"| {m} | {k} | {p['mean']:.4g} | {g['mean']:.4g} (n={g['n']}) | {d:.3g} | {allow:.3g} | {'pass' if ok else 'FAIL'} |")
L.append(f'\n**{"ALL PASS" if nfail == 0 else f"{nfail} FAIL: stop and send results/E0 back before running anything else"}**')
os.makedirs('results/E0', exist_ok=True); open('results/E0/repro.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[-1:])); print('written results/E0/repro.md')
