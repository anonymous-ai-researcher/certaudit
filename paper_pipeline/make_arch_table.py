"""paper/tab_app_arch.tex: every E2 architecture and width (GPU, 5 seeds), from the claims aggregation
gpu_results/E2/summary.csv; metrics over all proven pairs (these encoders read beyond the fanin cone), the
restructured Top-1 beside them, and EqAlign's retrieval on the encoder state h."""
import csv, json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
R = [r for r in csv.DictReader(open('gpu_results/E2/summary.csv')) if r['suite'] == 'epfl']
S = json.load(open('gpu_results/E2/selection.json'))
NAME = {'bi': 'Bidirectional', 'tf': 'Transformer'}
REG = [('Baseline', 'Baseline'), ('Augmentation', 'Augm.'), ('TT-sim', 'TT-sim'), ('Circuit contrast', 'Circuit c.'),
       ('L1 alignment', '$L_1$ align.'), ('L1 + TT-sim', '$L_1$ + TT'), ('EqAlign', '\\ours')]
L = ['\\begin{tabular}{ll c cc ccc}', '\\toprule',
     'Encoder & Regime & MAE & $\\rho_{\\text{all}}$ Syn. & $\\rho_{\\text{all}}$ Inj. & Top-1 all & Top-1 restr. & Top-1 H2 \\\\', '\\midrule']
archs = sorted({r['arch'] for r in R}, key=lambda a: (a.split('_')[0], int(a.split('_d')[1].split('_')[0])))
for a in archs:
    lam = S[a]['lam']
    for i, (key, nm) in enumerate(REG):
        rows = [r for r in R if r['arch'] == a and r['config'].startswith(key) and '[' not in r['config']
                and (key != 'TT-sim' or not r['config'].startswith('TT-sim (')) and (key != 'L1 alignment' or True)]
        rows = [r for r in rows if not (key == 'L1 alignment' and r['config'].startswith('L1 +'))]
        if key in ('TT-sim', 'Circuit contrast', 'L1 alignment', 'L1 + TT-sim', 'EqAlign'):
            m = {'TT-sim': 'ttsim_ph', 'Circuit contrast': 'circ_ph', 'L1 alignment': 'l1align', 'L1 + TT-sim': 'l1tt', 'EqAlign': 'eqalign_ph'}[key]
            rows = [r for r in rows if r['config'].endswith(f'lambda={lam[m]:g}')]
        assert len(rows) == 1, (a, key, [r['config'] for r in rows])
        f = lambda k: float(rows[0][k + '_mean'])
        lab = f'\\multirow{{7}}{{*}}{{{NAME[a[:2]]} {a.split("_d")[1].split("_")[0]}}}' if i == 0 else ''
        L.append(f"{lab} & {nm} & {f('MAE'):.4f} & {f('rho_allpairs_syn'):.2f} & {f('rho_allpairs_inj'):.2f} & {f('top1_all'):.1f} & "
                 f"{f('top1_seen'):.1f} & {f('top1_H2'):.1f} \\\\")
    h = [r for r in R if r['arch'] == a and r['config'] == f"EqAlign lambda={lam['eqalign_ph']:g} [eval h]"][0]
    L.append(f" & \\ours\\ on $h$ & -- & -- & -- & {float(h['top1_all_mean']):.1f} & {float(h['top1_seen_mean']):.1f} & {float(h['top1_H2_mean']):.1f} \\\\")
    L.append('\\midrule' if a != archs[-1] else '\\bottomrule')
L.append('\\end{tabular}')
open('paper/tab_app_arch.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[:14]))
