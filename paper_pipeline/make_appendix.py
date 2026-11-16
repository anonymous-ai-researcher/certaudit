"""Generate all appendix tables (paper/tab_app_*.tex) from result files."""
import json, pickle, glob, os, re, numpy as np
from analysis import load, group_summary, group_metrics, adv_matched, design_test, design_means, SYN, INJ
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # paths below are repo-relative

LAM = os.environ.get('LAMSTAR', '0.2'); TTL = os.environ.get('TTLAM', '0.1'); CL = os.environ.get('CIRCLAM', '0.2')
REG = [('Baseline', 'base'), ('Augmentation', 'aug'), ('TT-sim', f'ttsim_ph_l{TTL}'), ('Circuit contrast', f'circ_ph_l{CL}'),
       ('\\ours', f'eqalign_ph_l{LAM}')]
RAW = {n: [load(x[4:-4]) for x in sorted(glob.glob(f'raw_{p}_s*.pkl'))] for n, p in REG}
RAW = {n: v for n, v in RAW.items() if v}
FAMS = [('S1', ['S1'], None), ('S2', ['S2'], None), ('S3', ['S3'], None), ('S4', ['S4'], None),
        ('I.1', ["('inj', 0.1)"], None), ('I.3', ["('inj', 0.3)"], None), ('I.5', ["('inj', 0.5)"], None),
        ('H1', None, 'H1'), ('H2', None, 'H2')]
W = lambda name, L: open(f'paper/{name}.tex', 'w').write('\n'.join(L) + '\n')


def tab(caption, label, cols, header, rows, size='\\small'):
    L = ['\\begin{table}[h]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}', size,
         f'\\begin{{tabular}}{{{cols}}}', '\\toprule', header + ' \\\\', '\\midrule'] + rows + ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return L


# ---- pipeline / run time
def last_time(f):
    try:
        t = re.findall(r' (\d+)s', open(f).read()); return int(t[-1]) if t else None
    except FileNotFoundError:
        return None
rows = [
    'Cone extraction (EPFL) & \\texttt{data.py} & 11\\,s \\\\',
    'Fixed variants and certification (EPFL) & \\texttt{data.py} & 7\\,min \\\\',
    'Adversarial variants, 16 per circuit & \\texttt{evaluate.py} & 29\\,min \\\\',
    'SAT-proven matching of all EPFL pairs & \\texttt{rematch.py} & 43\\,s ($\\approx10^6$ SAT calls) \\\\',
    'Held-out variants H1, H2 (EPFL) & \\texttt{heldout\\_run.py} & 9\\,min \\\\',
    'ISCAS circuits, variants and matching & \\texttt{iscas\\_run.py} & see log \\\\',
]
for nm, f in [('Baseline', 'vlog_base__s0.log'), ('Augmentation', 'vlog_aug__s0.log'), ('\\ours', 'vlog_eqalign_ph_0.2_s0.log'),
              ('TT-sim', 'vlog_ttsim_0.1.log'), ('Circuit contrast', f'vlog_circ_{CL}.log'), ('Bidirectional baseline', 'vlog_bi_base_s0.log')]:
    t = last_time(f)
    if t:
        rows.append(f'Training: {nm} (12 epochs) & \\texttt{{train.py}} & {t // 60}\\,min \\\\')
W('tab_app_pipeline', tab('Pipeline stages and wall time on a 2-core CPU (training times measured with other jobs running concurrently).',
                          'tab:app_pipeline', 'lll', 'Stage & Script & Time', rows))

# ---- false matches per family
U = pickle.load(open('data_unverified.pkl', 'rb')); V = pickle.load(open('data.pkl', 'rb'))
rows = []
for lab, kinds, _ in FAMS[:7]:
    k = kinds[0]; a = b = c = 0
    for n in range(len(V['test'])):
        su = set(U['tvars'][n][k]['J'].tolist()); sv = set(V['tvars'][n][k]['J'].tolist()); p = V['tvars'][n][k]['var']['prob']
        fl = su - sv; a += len(su); b += len(fl); c += sum(1 for j in fl if min(p[j], 1 - p[j]) < 0.01)
    rows.append(f'{lab} & {a:,} & {b:,} & {100 * b / a:.2f} & {100 * c / max(b, 1):.0f} \\\\')
W('tab_app_false', tab('Simulation-only matching versus SAT (EPFL test set). Candidates: variant gates with a signature-matching original gate; false: no candidate is equivalent; near-constant: share of false matches with $\\min(p,1-p)<0.01$.',
                       'tab:app_false', 'lrrrr', 'Family & candidates & false & false (\\%) & near-constant (\\%)', rows))

# ---- validation
Vr = json.load(open('val_results.json'))
rows = []
for t in sorted(Vr):
    if t.startswith(('base_s0', 'aug_s0', 'eqalign', 'ttsim', 'circ', 'l1align', 'l1tt', 'bi_')) and t.endswith('_s0'):
        rows.append(f"\\texttt{{{t.replace('_', '\\_')}}} & {Vr[t]['val_mae']:.4f} & {100 * Vr[t]['val_top1_rs']:.1f} \\\\")
# baseline learning-rate candidates (GPU run E1a, seed 0); the selected one is base_s0 (2e-3)
_lr = 'work/gpu_results/E1/val_lr.json'
if os.path.exists(_lr):
    for t, v in sorted(json.load(open(_lr)).items()):
        if t.startswith('base_lr') and t != 'base_lr0.002_s0':
            rows.append(f"\\texttt{{{t.replace('_', '\\_')}}} & {v['val_mae']:.4f} & {100 * v['val_top1_rs']:.1f} \\\\")
W('tab_app_val', tab('All validation results (seed 0) used for selection; \\texttt{base\\_s0} uses the selected learning rate $2\\times10^{-3}$, \\texttt{base\\_lr} rows are the other two candidates.', 'tab:app_val', 'lrr', 'Run & val.\\ MAE & val.\\ Top-1 (\\%)', rows))

# ---- per family, all regimes (mean over seeds)
rows = []
for lab, kinds, ho in FAMS:
    for i, (n, Rs) in enumerate(RAW.items()):
        S = [group_summary(R, kinds=kinds, heldout=ho)[0] for R in Rs]
        g = lambda k, sc=1: np.mean([s[k] for s in S]) * sc
        rows.append(f"{lab if i == 0 else ''} & {n} & {g('mae_m_gap'):.4f} & {g('incons_rs'):.4f} & {g('rho'):.2f} & {g('top1_rs', 100):.1f} & {g('auroc_hard'):.3f} \\\\")
    rows.append('\\midrule' if lab != 'H2' else '')
rows = [r for r in rows if r]
W('tab_app_family', tab('Per-family results on EPFL (mean over 5 seeds). $\\bar\\delta_{\\mathrm{R}}$: absolute restructured inconsistency.',
                        'tab:app_family', 'llrrrrr', 'Family & Regime & $\\Delta$MAE & $\\bar\\delta_{\\mathrm{R}}$ & $\\rho$ & Top-1 (\\%) & AUROC$_\\mathrm{lvl}$', rows, '\\footnotesize'))

# ---- per design
designs = sorted(set(r['src'] for r in RAW['Baseline'][0]))
rows = []
for s in designs:
    cells = []
    for n, Rs in RAW.items():
        mae = np.mean([design_means(R, np.array([r['mae_orig'] for r in R]))[s] for R in Rs])
        top = np.mean([design_means(R, group_metrics(R, kinds=SYN + INJ)['top1_rs'])[s] for R in Rs])
        cells.append(f'{mae:.4f} / {100 * top:.1f}')
    nc = sum(r['src'] == s for r in RAW['Baseline'][0])
    rows.append(f"{s.split('/')[-1].replace('_', '\\_')} & {nc} & " + ' & '.join(cells) + ' \\\\')
W('tab_app_design', tab('Per test design: MAE on original circuits / restructured Top-1 (\\%), mean over seeds.', 'tab:app_design',
                        'lr' + 'c' * len(RAW), 'Design & circuits & ' + ' & '.join(RAW.keys()), rows, '\\footnotesize'))

# ---- design-level statistics
gm = lambda key: (lambda R: group_metrics(R, kinds=SYN + INJ)[key])
mae = lambda R: np.array([r['mae_orig'] for r in R])
rows = []
comps = [('\\ours', 'Baseline'), ('\\ours', 'Augmentation'), ('\\ours', 'TT-sim'), ('\\ours', 'Circuit contrast'), ('Augmentation', 'Baseline')]
for b_, a_ in comps:
    if a_ not in RAW or b_ not in RAW:
        continue
    for nm, f, sc, fmt in [('Top-1 (pts)', gm('top1_rs'), 100, '{:.1f}'), ('$\\bar\\delta_{\\mathrm{R}}$', gm('incons_rs'), 1, '{:.4f}'), ('MAE', mae, 1, '{:.4f}')]:
        r = design_test(RAW[a_], RAW[b_], f)
        rows.append(f"{b_} vs.\\ {a_} & {nm} & {fmt.format(sc * r['diff'])} & [{fmt.format(sc * r['lo'])}, {fmt.format(sc * r['hi'])}] & {r['n_pos']}/{r['n']} & {r['p_sign']:.3f} \\\\")
W('tab_app_stats', tab('Design-level comparisons (7 EPFL test designs; each design averaged over its circuits and 5 seeds). Diff.: B minus A; CI: 95\\% cluster bootstrap over designs; $n_+$: designs where B $>$ A; $p$: two-sided sign test.',
                       'tab:app_stats', 'llrcrr', 'B vs.\\ A & Metric & Diff. & 95\\% CI & $n_+$ & $p$', rows, '\\footnotesize'))

# ---- ISCAS per family
ISR = {n: [pickle.load(open(x, 'rb')) for x in sorted(glob.glob(f'rawis_{p}_s*.pkl'))] for n, p in REG}
ISR = {n: v for n, v in ISR.items() if v}
if ISR:
    rows = []
    for lab, kinds, ho in FAMS:
        for i, (n, Rs) in enumerate(ISR.items()):
            S = [group_summary(R, kinds=kinds, heldout=ho)[0] for R in Rs]
            g = lambda k, sc=1: np.mean([s[k] for s in S]) * sc
            rows.append(f"{lab if i == 0 else ''} & {n} & {g('mae_m_gap'):.4f} & {g('rho'):.2f} & {g('top1_rs', 100):.1f} \\\\")
        if lab != 'H2':
            rows.append('\\midrule')
    W('tab_app_iscas', tab('ISCAS85/89 test suite, per family (models trained on EPFL; mean over seeds).', 'tab:app_iscas', 'llrrr',
                           'Family & Regime & $\\Delta$MAE & $\\rho$ & Top-1 (\\%)', rows, '\\footnotesize'))

# ---- sweep
SW = {n: [x for f in sorted(glob.glob(f'sweep2_v_{p}_s*.json')) for x in json.load(open(f))] for n, p in REG}
SW = {n: v for n, v in SW.items() if v}
def red(rows_, pol, bits, field='wasted', emb=False):
    ref = vv = 0.0
    for key in sorted(set((x['circ'], x['kind'], x['model']) for x in rows_)):
        sel = [x for x in rows_ if (x['circ'], x['kind'], x['model']) == key and x['bits'] == bits]
        ref += [x[field] for x in sel if x['policy'] == 'topo'][0]
        vv += np.mean([x[field] + (x['emb_time'] if emb else 0) for x in sel if x['policy'] == pol])
    return 100 * (1 - vv / ref)
rows = []
bitsL = sorted(set(x['bits'] for x in SW['Baseline']))
anyr = SW['\\ours']
for lab, pol in [('random', 'random'), ('level heuristic', 'level'), ('true-match first', 'oracle')]:
    rows.append(f'{lab} & ' + ' & '.join(f'{red(anyr, pol, b):.0f}' for b in bitsL) + ' & -- \\\\')
for n, r_ in SW.items():
    rows.append(f'{n} embeddings & ' + ' & '.join(f'{red(r_, "emb", b):.0f}' for b in bitsL) + f' & {red(r_, "emb", 8, "time", True):.0f} \\\\')
W('tab_app_sweep', tab('SAT sweeping: reduction (\\%) in refuted SAT calls relative to topological order, per signature width; last column: change in total time at 8 bits including embedding inference, timed on the GPU machine while it trained other models (negative = slower; with inference on one CPU core, in the CPU run, \\ours\\ is \\SwCpuSlowEight\\% slower).',
                       'tab:app_sweep', 'l' + 'r' * (len(bitsL) + 1), 'Order & ' + ' & '.join(f'{b}-bit' for b in bitsL) + ' & time, 8-bit', rows, '\\footnotesize'))
beat = n_ = 0
for key in sorted(set((x['circ'], x['kind'], x['model'], x['bits']) for x in anyr)):
    sel = [x for x in anyr if (x['circ'], x['kind'], x['model'], x['bits']) == key]
    o = [x['wasted'] for x in sel if x['policy'] == 'oracle'][0]; e = [x['wasted'] for x in sel if x['policy'] == 'emb'][0]
    n_ += 1; beat += e < o
with open('paper/numbers.tex', 'a') as fh:
    fh.write(f'\\newcommand{{\\SwBeatTrue}}{{{beat}}}\n\\newcommand{{\\SwConfigs}}{{{n_}}}\n')
print('appendix tables written')
