"""E4 (data scale): 1, 2 or 4 certified variants per training cone, widths 64 and 256, all regimes, 5 seeds each
(server 2; learning rate and weights selected by the main-text rule within E4). Appends macros to paper/numbers.tex and
writes paper/tab_app_datascale.tex. The baseline uses no variants, so it has one row per width."""
import os, csv, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
S = [r for r in csv.DictReader(open('gpu_results/E4/summary.csv')) if r['suite'] == 'epfl' and '[' not in r['config']]
C = [r for r in csv.DictReader(open('gpu_results/E4/comparisons.csv')) if r['suite'] == 'epfl']
W = {'gru_d64_r2': '64', 'gru_d256_r2': '256'}
REG = [('base', 'Baseline'), ('aug', 'Augmentation'), ('ttsim_ph', 'TT-sim'), ('circ_ph', 'Circuit contrast'),
       ('l1align', '$L_1$ alignment'), ('l1tt', '$L_1$ alignment + TT-sim'), ('eqalign_ph', '\\ours')]


def get(a, m, v, k):
    r = [x for x in S if x['arch'] == a and x['method'] == m and x['variants'] == str(v)]
    assert len(r) == 1 and r[0]['n_seeds'] == '5', (a, m, v, len(r))
    return float(r[0][k + '_mean'])


M = {}
gain, gap, better = [], [], []
for a in W:
    gain.append(get(a, 'aug', 4, 'top1_seen') - get(a, 'aug', 1, 'top1_seen'))
    gap.append(get(a, 'eqalign_ph', 1, 'top1_seen') - get(a, 'aug', 4, 'top1_seen'))
    b = [r for r in C if r['arch'] == a and r['metric'] == 'top1_seen' and r['A'] == 'Augmentation V=1' and r['B'] == 'Augmentation V=4']
    better.append(b[0]['designs_better'])
M['EFourAugGainMin'] = f'{min(gain):.1f}'; M['EFourAugGainMax'] = f'{max(gain):.1f}'
M['EFourGapMin'] = f'{min(gap):.1f}'; M['EFourGapMax'] = f'{max(gap):.1f}'
M['EFourAugBetter'] = ' and '.join(sorted(set(better)))
oursd = [abs(get(a, 'eqalign_ph', 4, 'top1_seen') - get(a, 'eqalign_ph', 1, 'top1_seen')) for a in W]
M['EFourOursChangeMax'] = f'{max(oursd):.1f}'
rs = [get(a, m, v, 'rho_syn') for a in W for m, _ in REG[1:] for v in (1, 2, 4)]
# H2 fails for every regime, width and variant count (rule of the claims check: dMAE_H2 >= 0.05 and rho_H2 >= 2)
cl = open('gpu_results/E4/claims.md').read()
assert cl.count('**PASS (all fail on H2)**') == cl.count('C4 no tested repair survives H2') == 6, 'H2 check'
h2 = [get(a, m, v, 'top1_H2') for a in W for m, _ in REG for v in ((1,) if m == 'base' else (1, 2, 4))]
M['EFourHTwoTopMax'] = f'{max(h2):.1f}'

L = ['\\begin{tabular}{l ccc cc ccc cc}', '\\toprule',
     ' & \\multicolumn{5}{c}{Width 64} & \\multicolumn{5}{c}{Width 256} \\\\', '\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}',
     ' & \\multicolumn{3}{c}{Top-1 (\\%)} & \\multicolumn{2}{c}{$\\rho$ Syn.} & \\multicolumn{3}{c}{Top-1 (\\%)} & \\multicolumn{2}{c}{$\\rho$ Syn.} \\\\',
     '\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}',
     'Regime & $V{=}1$ & 2 & 4 & 1 & 4 & 1 & 2 & 4 & 1 & 4 \\\\', '\\midrule']
for m, nm in REG:
    row = [nm]
    for a in W:
        if m == 'base':
            row += [f"{get(a, m, 1, 'top1_seen'):.1f}", '--', '--', f"{get(a, m, 1, 'rho_syn'):.2f}", '--']
        else:
            row += [f"{get(a, m, v, 'top1_seen'):.1f}" for v in (1, 2, 4)] + [f"{get(a, m, v, 'rho_syn'):.2f}" for v in (1, 4)]
    L.append(' & '.join(row) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}']
open('paper/tab_app_datascale.tex', 'w').write('\n'.join(L) + '\n')
with open('paper/numbers.tex', 'a') as fh:
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v.replace("-", chr(92) + "ensuremath{-}")}}}\n')
print(M); print('\n'.join(L))
