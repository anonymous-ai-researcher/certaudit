"""Generate Table V (audit of public checkpoints) and its prose numbers from the
external-audit result files produced by ext_audit/run_dg2.py and run_dg3.py."""
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, os
sys.path.insert(0, 'ext_audit')
from summarize_ext import summarize, summarize_mean
# PI protocol: DeepGate2 draws random primary-input vectors per forward pass; the paper reports the protocol in
# which a circuit and its variants share them (ext_audit/sharedpi, convert_shared.py). EXT_PROTOCOL=random gives
# the original audit files.
EXTDIR = 'ext_audit/' if os.environ.get('EXT_PROTOCOL') == 'random' else 'ext_audit/sharedpi/'
import glob

# LaTeX macro names cannot contain digits, so Top-1 columns get spelled-out names.
SAFE = {'mae': 'Mae', 'dmae_inj': 'DmaeInj', 'rho_syn': 'RhoSyn', 'rho_inj': 'RhoInj',
        'rho_ho': 'RhoHo', 'top1_seen': 'TopSeen', 'top1_ho': 'TopHo',
        'auroc_hard': 'AurocHard', 'worst_ratio': 'WorstRatio', 'chance': 'Chance'}
ROWS = [('DeepGate2~\\cite{deepgate2}', EXTDIR + 'results_deepgate2.json', 'DTwo'),
        ('DeepGate3~\\cite{deepgate3}', EXTDIR + 'results_deepgate3.json', 'DThree'),
        ('DeepGate2, fine-tuned', EXTDIR + 'results_dg2ft_control_s*.json', 'FtCtrl'),
        ('DeepGate2, fine-tuned with \\ours', EXTDIR + 'results_dg2ft_eqalign_s*.json', 'FtEq')]
L = ['\\begin{tabular}{l c c c c c c}', '\\toprule',
     'Model & MAE & $\\Delta$MAE inj. & $\\rho$ syn. & $\\rho$ h.-o. & Top-1 seen & Top-1 h.-o. \\\\',
     '\\midrule']
M = {}
for name, path, pref in ROWS:
    if not glob.glob(path):
        print('missing', path); continue
    d = summarize_mean(path) if '*' in path else summarize(path)
    if '*' in path: M[pref + 'Seeds'] = str(d['n_seeds'])
    L.append(f"{name} & {d['mae']:.4f} & {d['dmae_inj']:+.4f} & {d['rho_syn']:.2f} & "
             f"{d['rho_ho']:.2f} & {d['top1_seen']:.1f}\\% & {d['top1_ho']:.1f}\\% \\\\")
    for k, fmt, sc in [('mae', '{:.4f}', 1), ('dmae_inj', '{:+.4f}', 1), ('rho_syn', '{:.2f}', 1),
                       ('rho_inj', '{:.2f}', 1), ('rho_ho', '{:.2f}', 1), ('top1_seen', '{:.1f}', 1),
                       ('top1_ho', '{:.1f}', 1), ('auroc_hard', '{:.3f}', 1), ('worst_ratio', '{:.2f}', 1),
                       ('chance', '{:.1f}', 1)]:
        M[pref + SAFE[k]] = fmt.format(d[k] * sc)
L += ['\\bottomrule', '\\end{tabular}']
# DeepGate2 -> DeepGate3: how much the error and the absolute disagreement fall under synthesis.
if all(glob.glob(p) for _, p, _ in ROWS):
    d2, d3 = summarize(ROWS[0][1]), summarize(ROWS[1][1])
    if glob.glob(ROWS[3][1]):
        fc, fe = summarize_mean(ROWS[2][1]), summarize_mean(ROWS[3][1])
        M['FtMaeCostPct'] = f"{100 * (fe['mae'] / fc['mae'] - 1):.0f}"
    M['DThreeErrDrop'] = f"{100 * (1 - d3['err_syn'] / d2['err_syn']):.0f}"
    M['DThreeInconsDrop'] = f"{100 * (1 - d3['incons_syn'] / d2['incons_syn']):.0f}"
if os.path.exists('ext_audit/full_pi/selfcheck.json'):    # same circuit against itself, all 376 test circuits
    sc = json.load(open('ext_audit/full_pi/selfcheck.json'))
    M['DTwoSelfRand'] = f"{100 * sc['dg2_random']:.0f}"; M['DTwoSelfShared'] = f"{100 * sc['dg2_shared']:.0f}"
    M['DThreeSelfRand'] = f"{100 * sc['dg3_random']:.0f}"; M['DThreeSelfShared'] = f"{100 * sc['dg3_shared']:.0f}"
# E11: released MGVGA and TRACE (no probability head): restructured Top-1 (pessimistic ties) on seen families,
# H1 and H2, and the self-retrieval ceiling (ext_audit/new/eval_ext2.py)
for pre, f in (('Mg', 'ext_audit/new/results2_mgvga.json'), ('Tr', 'ext_audit/new/results2_trace.json'), ('Fg', 'ext_audit/new/results2_fgnn2.json')):
    if os.path.exists(f):
        S = json.load(open(f))['summary']
        M[pre + 'Seen'] = f"{100 * S['seen']['top1_rs']:.1f}"; M[pre + 'HOne'] = f"{100 * S['H1']['top1_rs']:.1f}"
        M[pre + 'HTwo'] = f"{100 * S['H2']['top1_rs']:.1f}"; M[pre + 'Self'] = f"{100 * S['self']['top1']:.0f}"
        M[pre + 'SeenR'] = f"{100 * S['seen']['top1r_rs']:.1f}"; M[pre + 'HTwoR'] = f"{100 * S['H2']['top1r_rs']:.1f}"
        M[pre + 'SelfR'] = f"{100 * S['self']['top1r']:.0f}"
for pre, f in (('DTwo', EXTDIR + 'results_deepgate2.json'), ('DThree', EXTDIR + 'results_deepgate3.json')):
    if os.path.exists(f):
        r = json.load(open(f)); M[pre + 'HTwoRS'] = f"{100 * r['H2']['top1_rs']:.1f}"
with open('paper/numbers.tex', 'a') as fh:
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('external macros:', len(M))
