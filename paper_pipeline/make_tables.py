"""Generate paper/tab_ablation.tex (seed 0) from validation and test results, revised metrics."""
import json, os
import numpy as np
from analysis import load, group_summary, SYN, INJ

V = json.load(open('val_results.json'))
LAM = os.environ.get('LAMSTAR', '0.2'); TTL = os.environ.get('TTLAM', '0.1')
# weight grids fixed before training (GPU run: tools/plan.py GRIDS; CPU run: the defaults)
EQG = os.environ.get('EQ_GRID', '0.05,0.1,0.2,0.5').split(','); TTG = os.environ.get('TT_GRID', '0.1,0.5,1,2').split(',')
CG = os.environ.get('C_GRID', '0.2,1').split(',')
rows = [('Baseline', 'base_s0'), ('Augmentation', 'aug_s0'), None] + \
       [(f'\\ours\\ $\\lambda$={l}' + ('$^\\star$' if l == LAM else ''), f'eqalign_ph_l{l}_s0') for l in EQG[:4]] + \
       [('\\ours\\ no proj., $\\lambda$=0.2', 'eqalign_l0.2_s0'), None] + \
       [(f'TT-sim $\\lambda_{{\\mathrm{{TT}}}}$={TTL}$^\\star$', f'ttsim_ph_l{TTL}_s0'),
        ('Circuit contrast $\\lambda_{\\mathrm{C}}$=0.2$^\\star$', 'circ_ph_l0.2_s0')]
FULL = rows[:2] + [None] + [(f'\\ours\\ $\\lambda$={l}' + ('$^\\star$' if l == LAM else ''), f'eqalign_ph_l{l}_s0') for l in EQG] + \
       [('\\ours\\ no proj., $\\lambda$=0.2', 'eqalign_l0.2_s0'), None] + \
       [(f'TT-sim $\\lambda_{{\\mathrm{{TT}}}}$={l}' + ('$^\\star$' if l == TTL else ''), f'ttsim_ph_l{l}_s0') for l in TTG] + [None] + \
       [(f'Circuit contrast $\\lambda_{{\\mathrm{{C}}}}$={l}' + ('$^\\star$' if l == '0.2' else ''), f'circ_ph_l{l}_s0') for l in CG]
# the two MixGate-style regimes (positives-only L1 alignment on h, alone or with the TT-sim term), same rule
L1G = os.environ.get('L1_GRID', '0.0003,0.001,0.003,0.01,0.03').split(','); L1L = os.environ.get('L1LAM', '0.0003')
L1TG = os.environ.get('L1TT_GRID', '0.001,0.003,0.01').split(','); L1TL = os.environ.get('L1TTLAM', '0.003')
FULL += [None] + [(f'$L_1$ alignment $\\lambda$={l}' + ('$^\\star$' if l == L1L else ''), f'l1align_l{l}_s0') for l in L1G] + \
        [None] + [(f'$L_1$ alignment + TT-sim $\\lambda$={l}' + ('$^\\star$' if l == L1TL else ''), f'l1tt_l{l}_s0') for l in L1TG]
def make(rows, out):
    L = ['\\begin{tabular}{l cc cccc}', '\\toprule',
         ' & \\multicolumn{2}{c}{Validation} & \\multicolumn{4}{c}{Test (seen families)} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}',
         'Variant & MAE & Top-1 & MAE & $\\rho$ & Top-1 & AUROC$_\\mathrm{lvl}$ \\\\', '\\midrule']
    for row in rows:
        if row is None:
            L.append('\\midrule'); continue
        name, tag = row
        if not os.path.exists(f'raw_{tag}.pkl') or tag not in V:
            L.append(f'{name} & \\multicolumn{{6}}{{c}}{{(pending)}} \\\\'); continue
        R = load(tag); s = group_summary(R, kinds=SYN + INJ)[0]; v = V[tag]
        L.append(f"{name} & {v['val_mae']:.4f} & {100*v['val_top1_rs']:.1f} & {np.mean([r['mae_orig'] for r in R]):.4f} & "
                 f"{s['rho']:.2f} & {100*s['top1_rs']:.1f} & {s['auroc_hard']:.3f} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open(out, 'w').write('\n'.join(L) + '\n')
    return L

make(rows, 'paper/tab_ablation.tex'); make(FULL, 'paper/tab_ablation_full.tex')
