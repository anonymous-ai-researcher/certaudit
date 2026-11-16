"""paper/tab_stats.tex: design-level comparisons, one column, for the main text."""
import glob, pickle, os
import numpy as np
from analysis import group_metrics, design_test, SYN, INJ

LAM = os.environ.get('LAMSTAR', '0.2'); TTL = os.environ.get('TTLAM', '0.1'); CL = os.environ.get('CIRCLAM', '0.2')
REG = [('Baseline', 'base'), ('Augmentation', 'aug'), ('TT-sim', f'ttsim_ph_l{TTL}'),
       ('Circuit contrast', f'circ_ph_l{CL}'), ('\\ours', f'eqalign_ph_l{LAM}')]
RAW = {n: [pickle.load(open(x, 'rb')) for x in sorted(glob.glob(f'raw_{p}_s*.pkl'))] for n, p in REG}
RAW = {n: v for n, v in RAW.items() if v}
gm = lambda key: (lambda R: group_metrics(R, kinds=SYN + INJ)[key])
mae = lambda R: np.array([r['mae_orig'] for r in R])

L = ['\\begin{tabular}{l rc rc rc}', '\\toprule',
     ' & \\multicolumn{2}{c}{Top-1 (pts) $\\uparrow$} & \\multicolumn{2}{c}{$\\bar\\delta_{\\mathrm{R}}$ $\\downarrow$}'
     ' & \\multicolumn{2}{c}{MAE $\\downarrow$} \\\\',
     '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
     '\\ours\\ vs. & diff. & $n_+$ & diff. & $n_+$ & diff. & $n_+$ \\\\', '\\midrule']
for a_ in ['Baseline', 'Augmentation', 'TT-sim', 'Circuit contrast']:
    if a_ not in RAW:
        continue
    cells = []
    # n_better counts the designs on which EqAlign is better, whichever direction "better" is:
    # higher for Top-1, lower for the inconsistency and for MAE. (design_test's own n_pos counts
    # a positive difference, which for the two lower-is-better columns means EqAlign is worse.)
    for f, sc, fmt, lower in [(gm('top1_rs'), 100, '{:.1f}', False), (gm('incons_rs'), 1, '{:.4f}', True),
                              (mae, 1, '{:.4f}', True)]:
        r = design_test(RAW[a_], RAW['\\ours'], f)
        diff = np.array(r['b']) - np.array(r['a'])
        n_better = int((diff < 0).sum()) if lower else int((diff > 0).sum())
        d = fmt.format(sc * r['diff']).replace('-', '\\ensuremath{-}')
        star = '$^\\star$' if r['p_sign'] < 0.05 else ''
        cells += [f'{d}{star}', f"{n_better}/{r['n']}"]
    L.append(f"{a_} & " + ' & '.join(cells) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}']
open('paper/tab_stats.tex', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
