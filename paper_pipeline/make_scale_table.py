"""paper/tab_app_scale.tex: blindness and repair against encoder size (GPU, 5 seeds each), from the claims
aggregations of E1a (recurrent 64, 256), E1b (recurrent 128), E1c (recurrent 256 with 4 rounds), E2 (transformer 128, 256) and E15 (recurrent 1024, transformer 512)."""
import csv, json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
NP = json.load(open('gpu_results/n_params.json'))
SC = [('E1', 'gru_d64_r2', 'Recurrent 64'), ('E1', 'gru_d128_r2', 'Recurrent 128'), ('E1', 'gru_d256_r2', 'Recurrent 256'), ('E1c', 'gru_d256_r4', 'Recurrent 256, 4 rounds'), ('E15', 'gru_d1024_r2', 'Recurrent 1024'),
      ('E2', 'tf_d128_r2', 'Transformer 128'), ('E2', 'tf_d256_r2', 'Transformer 256'), ('E15', 'tf_d512_r2', 'Transformer 512')]
L = ['\\begin{tabular}{l r cc ccc c}', '\\toprule',
     ' & & \\multicolumn{2}{c}{Baseline} & \\multicolumn{3}{c}{Top-1 (\\%)} & \\\\',
     '\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}',
     'Encoder & Params & $\\rho$ Syn. & MAE & Base & Augm. & \\ours & $\\rho_{\\text{H2}}$ (all) \\\\', '\\midrule']
for e, arch, nm in SC:
    R = {r['config']: r for r in csv.DictReader(open(f'gpu_results/{e}/summary.csv')) if r['suite'] == 'epfl' and r['arch'] == arch}
    tf = arch.startswith('tf'); met = 'top1_all' if tf else 'top1_seen'; rk = 'rho_allpairs_syn' if tf else 'rho_syn'
    ours = [c for c in R if c.startswith('EqAlign') and '[' not in c][0]
    regs = ['Baseline', 'Augmentation', ours]
    h2 = [float(R[c]['rho_H2_mean']) for c in regs]
    f = lambda c, k: float(R[c][k + '_mean'])
    p = NP[arch]; ps = f'{p / 1e6:.1f}M' if p >= 1e6 else f'{p / 1e3:.0f}k'
    L.append(f"{nm} & {ps} & {f('Baseline', rk):.2f} & {f('Baseline', 'MAE'):.4f} & {f('Baseline', met):.1f} & "
             f"{f('Augmentation', met):.1f} & {f(ours, met):.1f} & {min(h2):.1f} to {max(h2):.1f} \\\\")
    if arch == 'gru_d1024_r2':
        L.append('\\midrule')
L += ['\\bottomrule', '\\end{tabular}']
open('paper/tab_app_scale.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
