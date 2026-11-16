"""Extended report: the public-checkpoint audit under both primary-input protocols (paper/tab_pi_protocol.tex)."""
import os, sys, json
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, 'ext_audit')
from summarize_ext import summarize, summarize_mean
sc = json.load(open('ext_audit/full_pi/selfcheck.json'))
R = [('DeepGate2', 'results_deepgate2.json', 'dg2'), ('DeepGate3 tok.', 'results_deepgate3.json', 'dg3'),
     ('DG2 + ft.', 'results_dg2ft_control_s*.json', None), ('DG2 + \\ours', 'results_dg2ft_eqalign_s*.json', None)]
L = ['\\begin{tabular}{ll cccccc c}', '\\toprule',
     'Model & PI vectors & MAE & $\\rho$ syn. & $\\rho$ inj. & $\\rho$ h.-o. & Top-1 seen & Top-1 h.-o. & self Top-1 \\\\', '\\midrule']
for name, f, k in R:
    for proto, d in (('random (released)', 'ext_audit/'), ('shared (paper)', 'ext_audit/sharedpi/')):
        s = summarize_mean(d + f) if '*' in f else summarize(d + f)
        st = f"{100 * sc[k + ('_random' if proto.startswith('random') else '_shared')]:.1f}" if k else '--'
        L.append(f"{name if proto.startswith('random') else ''} & {proto} & {s['mae']:.4f} & {s['rho_syn']:.2f} & {s['rho_inj']:.2f} & "
                 f"{s['rho_ho']:.2f} & {s['top1_seen']:.1f} & {s['top1_ho']:.1f} & {st} \\\\")
    L.append('\\addlinespace')
L += ['\\bottomrule', '\\end{tabular}']
open('paper/tab_pi_protocol.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
