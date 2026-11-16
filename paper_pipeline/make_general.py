"""paper/tab_general.tex from numbers.json, plus the public-checkpoint block from the
external-audit result files (ext_audit/results_deepgate{2,3}.json)."""
import json, os, sys, glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'ext_audit')
M = json.load(open('numbers.json'))
def row(label, pre, k, name, color, hatch=False):
    g = lambda key: M.get(f'{pre}{k}{key}', '--')
    sq = f'\\mk{{{color}}}'
    return f"{label} & {sq}\\ {name} & {g('MAE')} & {g('RhoSyn')} & {g('RhoInj')} & {g('RhoHo')} & {g('DHo')} & {g('TopRS')} & {g('TopHo')} \\\\"
L = ['\\begin{tabular}{ll c ccc c cc}', '\\toprule',
     ' & & MAE$\\downarrow$ & \\multicolumn{3}{c}{$\\rho\\downarrow$} & $\\Delta$MAE$\\downarrow$ & \\multicolumn{2}{c}{Top-1 (\\%)$\\uparrow$} \\\\',
     '\\cmidrule(lr){4-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-9}',
     'Setting & Model & orig. & Syn. & Inj. & H.-o. & H.-o. & seen & H.-o. \\\\', '\\midrule']
models = [('Base', 'Baseline', 'cbase', False), ('Aug', 'Augmentation', 'caug', False), ('TT', 'TT-sim', 'ctt', False),
          ('Circ', 'Circuit contrast', 'ccirc', True), ('Ours', '\\ours', 'cours', False)]
for i, (k, nm, c, h) in enumerate(models):
    if f'IS{k}RhoSyn' in M:
        L.append(row('\\multirow{5}{*}{ISCAS}' if i == 0 else '', 'IS', k, nm, c, h))
L.append('\\midrule')
for _pre, _lab in (('Bi', '\\shortstack[l]{Bidir.\\\\$w{=}64$}'), ('Tf', '\\shortstack[l]{Transf.\\\\$w{=}128$}')):
    bi = [m for m in models if f'{_pre}{m[0]}RhoSyn' in M]
    if not bi:
        continue
    if _pre == 'Tf':
        L.append('\\midrule')
    for i, (k, nm, c, h) in enumerate(bi):
        L.append(row(f'\\multirow{{{len(bi)}}}{{*}}{{{_lab}}}' if i == 0 else '', _pre, k, nm, c, h))
# Public checkpoints: the same question as ISCAS and Bi, asked of models we did not train.
from summarize_ext import summarize, summarize_mean
# PI protocol: DeepGate2 draws random primary-input vectors per forward pass; the paper reports the protocol in
# which a circuit and its variants share them (ext_audit/sharedpi, convert_shared.py). EXT_PROTOCOL=random gives
# the original audit files.
EXTDIR = 'ext_audit/' if os.environ.get('EXT_PROTOCOL') == 'random' else 'ext_audit/sharedpi/'
EXT = [('DeepGate2', EXTDIR + 'results_deepgate2.json'), ('DG3 tokenizer', EXTDIR + 'results_deepgate3.json'),
       ('DG2 + own loss', EXTDIR + 'results_dg2ft_control_s*.json'), ('DG2 + \\ours', EXTDIR + 'results_dg2ft_eqalign_s*.json')]
ext = [(nm, summarize_mean(p) if '*' in p else summarize(p)) for nm, p in EXT if glob.glob(p)]
# Fine-tuned DeepGate2: the 5-seed GPU rerun (E3, the same configuration: 5 epochs, DeepGate2's own random primary
# inputs in training, shared-input evaluation) replaces the 2-seed CPU run when its results are present
import csv
_g = 'gpu_results/E3/summary.csv'
if os.path.exists(_g):
    _cf = {'DG2 + own loss': 'DG2 fine-tuned (control), 5 ep, train_pi=random',
           'DG2 + \\ours': 'DG2 fine-tuned + EqAlign lambda=0.2, 5 ep, train_pi=random'}
    _R = {r['config']: r for r in csv.DictReader(open(_g)) if r['suite'] == 'epfl'}
    def _gpu(c):
        f = lambda k: float(_R[c][k + '_mean'])
        return dict(mae=f('MAE'), rho_syn=f('rho_syn'), rho_inj=f('rho_inj'), rho_ho=(f('rho_H1') + f('rho_H2')) / 2,
                    dmae_ho=(f('dMAE_H1') + f('dMAE_H2')) / 2, top1_seen=f('top1_seen'), top1_ho=f('top1_ho'))
    ext = [(nm, _gpu(_cf[nm]) if nm in _cf else d) for nm, d in ext]
# DeepGate2 retrained from its released code and data (E10, 3 seeds; shared-input evaluation like the released row)
_g10 = 'gpu_results/E10/summary.csv'
if os.path.exists(_g10):
    _r = [r for r in csv.DictReader(open(_g10)) if r['suite'] == 'epfl' and '[' not in r['config']][0]
    f = lambda k: float(_r[k + '_mean'])
    ext.insert(1, ('DG2 retrained$^\\dagger$', dict(mae=f('MAE'), rho_syn=f('rho_syn'), rho_inj=f('rho_inj'), rho_ho=(f('rho_H1') + f('rho_H2')) / 2,
                                        dmae_ho=(f('dMAE_H1') + f('dMAE_H2')) / 2, top1_seen=f('top1_seen'), top1_ho=f('top1_ho'))))
# Encoders without a probability head (MGVGA, TRACE): representation metrics only, same pairs and the same
# pessimistic-tie Top-1 (ext_audit/new/eval_ext2.py); H.-o. is the mean of H1 and H2 as for the other rows.
for nm, f in (('TRACE', 'ext_audit/new/results2_trace.json'), ('MGVGA', 'ext_audit/new/results2_mgvga.json'), ('FGNN2', 'ext_audit/new/results2_fgnn2.json')):
    if os.path.exists(f):
        S = json.load(open(f))['summary']
        ext.insert(len([e for e in ext if e[0].startswith(('DeepGate', 'DG2 retrained', 'DG3'))]), (nm, dict(
            mae=None, top1_seen=100 * S['seen']['top1_rs'], top1_ho=100 * (S['H1']['top1_rs'] + S['H2']['top1_rs']) / 2)))
npub = len([e for e in ext if not e[0].startswith('DG2 +')])
GROUPS = [('\\shortstack[l]{Public\\\\(CPU)}', ext[:npub]), ('\\shortstack[l]{FT\\\\(GPU)}', ext[npub:])]
for g, rows in GROUPS:
    if not rows: continue
    L.append('\\midrule')
    for i, (nm, d) in enumerate(rows):
        lab = f'\\multirow{{{len(rows)}}}{{*}}{{{g}}}' if i == 0 else ''
        sw = '\\mk{cours}\\ ' if 'ours' in nm else ''
        if d['mae'] is None:
            L.append(f"{lab} & {sw}{nm} & -- & -- & -- & -- & -- & {d['top1_seen']:.1f} & {d['top1_ho']:.1f} \\\\")
            continue
        L.append(f"{lab} & {sw}{nm} & {d['mae']:.4f} & {d['rho_syn']:.2f} & {d['rho_inj']:.2f} & {d['rho_ho']:.2f} & "
                 f"{d['dmae_ho']:.4f} & {d['top1_seen']:.1f} & {d['top1_ho']:.1f} \\\\")
L += ['\\bottomrule', '\\end{tabular}']
open('paper/tab_general.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
