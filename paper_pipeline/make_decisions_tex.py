"""Decision consequences on the main-text models (E16: GPU, width 64, 5 seeds, E1a weights; nothing trained or tuned).
Reads gpu_results/E16/decisions.json (decisions.py on gpu_results/E16/pairpred) and the pair predictions; appends
macros to paper/numbers.tex and writes paper/tab_app_decisions.tex.
Rules fixed in advance (design note): CASCAD-style phase hint, tau in {0.005, 0.01, 0.05}; hard-to-activate
flag p < 0.05 or p > 0.95; top-5% hardest gates per circuit and family, Jaccard; restructured pairs."""
import os, glob, json, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open('gpu_results/E16/decisions.json'))
P, T = D['pooled'], D['tests']
pc = lambda x: f'{100 * x:.1f}'
M = {}
b = P['syn']['Base']
assert D['rule']['passes'] and all(v == 5 for v in D['n_seeds'].values())
M['DecFlipSyn'] = pc(b['flip_0.01'][0]); M['DecErrSyn'] = pc(b['err_0.01'][0])
M['DecRatioSyn'] = f"{100 * b['flip_0.01'][0] / b['err_0.01'][0]:.0f}"
M['DecCFlipSyn'] = f"{100 * b['cflip_0.01'][0]:.0f}"
h = P['H2']['Base']
M['DecFlipHTwo'] = pc(h['flip_0.01'][0]); M['DecErrHTwo'] = pc(h['err_0.01'][0]); M['DecCFlipHTwo'] = f"{100 * h['cflip_0.01'][0]:.0f}"
i = P['inj']['Base']
M['DecFlipInj'] = pc(i['flip_0.01'][0]); M['DecErrInj'] = pc(i['err_0.01'][0]); M['DecCFlipInj'] = f"{100 * i['cflip_0.01'][0]:.0f}"
for g, k in (('syn', 'Syn'), ('inj', 'Inj'), ('H2', 'HTwo')):
    M[f'DecJac{k}'] = f"{P[g]['Base']['jacc'][0]:.2f}"
# at CASCAD's thresholds no regime ever hints opposite phases for a proven pair (at 0.05: at most 0.02% of pairs)
assert max(P[g][r][f'opp_{t}'][0] for g in ('syn', 'inj', 'H2') for r in D['regimes'] for t in ('0.005', '0.01')) == 0
assert max(P[g][r]['opp_0.05'][0] for g in ('syn', 'inj', 'H2') for r in D['regimes']) < 0.0003
# robustness of the rule over the thresholds fixed in advance (baseline, synthesis)
rat = [b[f'flip_{t}'][0] / b[f'err_{t}'][0] for t in ('0.005', '0.01', '0.05')]
M['DecRatioMin'] = f'{100 * min(rat):.0f}'; M['DecRatioMax'] = f'{100 * max(rat):.0f}'
# regimes against the baseline (design level, exact sign-flip test)
for r, k in (('Aug', 'Aug'), ('TT', 'TT'), ('Ours', 'Ours')):
    t = T['inj'][r]['flip_0.01']
    M[f'Dec{k}InjDiff'] = f"{100 * t['diff']:.1f}"; M[f'Dec{k}InjBetter'] = t['better']; M[f'Dec{k}InjP'] = f"{t['p']:.3f}"
    s = T['syn'][r]['flip_0.01']; M[f'Dec{k}SynBetter'] = s['better']
# the appendix states one count and one p for augmentation, TT-sim and EqAlign under injection: they must agree
assert len({(M[f'Dec{k}InjBetter'], M[f'Dec{k}InjP']) for k in ('Aug', 'TT', 'Ours')}) == 1
# hint rates: EqAlign hints fewer gates (predictions further from 0 and 1), so it misses more true hints
G = {'syn': ['S1', 'S2', 'S3', 'S4']}
for pre, k in (('base', 'Base'), ('eqalign_ph_l0.2', 'Ours')):
    hint, miss = [], []
    for f in sorted(glob.glob(f'gpu_results/E16/pairpred/{pre}_s*.npz')):
        z = np.load(f, allow_pickle=False); fams = list(z['fams'])
        m = np.isin(z['fam'], [fams.index(x) for x in G['syn']]) & z['rs']
        pv, p = z['pv'][m], z['p'][m]
        cv = (pv < 0.01) | (pv > 0.99); ct = (p < 0.01) | (p > 0.99)
        hint.append(cv.mean()); miss.append((ct & ~cv).mean())
    M[f'Dec{k}HintSyn'] = pc(np.mean(hint)); M[f'Dec{k}MissSyn'] = pc(np.mean(miss))
M['DecTrueHintSyn'] = pc(P['syn']['_truecommit']['0.01'])

# appendix table: regime x rewrite group, tau = 0.01
NM = {'Base': 'Baseline', 'Aug': 'Augmentation', 'TT': 'TT-sim', 'Circ': 'Circuit contrast', 'Ours': '\\ours'}
L = ['\\begin{tabular}{l ccc ccc ccc}', '\\toprule',
     ' & \\multicolumn{3}{c}{Synthesis} & \\multicolumn{3}{c}{Injection} & \\multicolumn{3}{c}{H2} \\\\',
     '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
     'Regime & Flip & Err. & Jacc. & Flip & Err. & Jacc. & Flip & Err. & Jacc. \\\\', '\\midrule']
for r in D['regimes']:
    row = [NM[r]]
    for g in ('syn', 'inj', 'H2'):
        Q = P[g][r]; row += [pc(Q['flip_0.01'][0]), pc(Q['err_0.01'][0]), f"{Q['jacc'][0]:.2f}"]
    L.append(' & '.join(row) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}']
open('paper/tab_app_decisions.tex', 'w').write('\n'.join(L) + '\n')
with open('paper/numbers.tex', 'a') as fh:
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v.replace("-", chr(92) + "ensuremath{-}")}}}\n')
print(M); print('\n'.join(L))
