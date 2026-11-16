"""Macros from the GPU experiments run on the 8-GPU machine (gpu_results/<exp>/summary.csv and comparisons.csv,
copied unchanged from the returned result snapshots). Appended to paper/numbers.tex by build.sh."""
import csv, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
G = 'gpu_results'
def rows(exp, f):
    p = f'{G}/{exp}/{f}.csv'
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def s(exp, arch, config, key):
    for r in rows(exp, 'summary'):
        if r['suite'] == 'epfl' and r['arch'] == arch and r['config'] == config:
            return float(r[key + '_mean'])
def c(exp, arch, B, A, metric):
    for r in rows(exp, 'comparisons'):
        if r['suite'] == 'epfl' and r['arch'] == arch and r['B'] == B and r['A'] == A and r['metric'] == metric:
            return r
M = {}
for tag, arch in (('', 'gru_d64_r2'), ('W', 'gru_d256_r2')):
    eq, eqh = s('E1', arch, 'EqAlign lambda=0.2', 'top1_H2'), s('E9', arch, 'EqAlign lambda=0.2, h2-trained', 'top1_H2')
    if eq is None or eqh is None:
        continue
    M[f'NineEq{tag}HTwoBefore'] = f'{eq:.1f}'; M[f'NineEq{tag}HTwoAfter'] = f'{eqh:.1f}'
    M[f'NineEq{tag}SeenBefore'] = f"{s('E1', arch, 'EqAlign lambda=0.2', 'top1_seen'):.1f}"
    M[f'NineEq{tag}SeenAfter'] = f"{s('E9', arch, 'EqAlign lambda=0.2, h2-trained', 'top1_seen'):.1f}"
    M[f'NineEq{tag}DHTwoBefore'] = f"{s('E1', arch, 'EqAlign lambda=0.2', 'dMAE_H2'):.3f}"
    M[f'NineEq{tag}DHTwoAfter'] = f"{s('E9', arch, 'EqAlign lambda=0.2, h2-trained', 'dMAE_H2'):.3f}"
    m0, m1 = s('E1', arch, 'EqAlign lambda=0.2', 'MAE'), s('E9', arch, 'EqAlign lambda=0.2, h2-trained', 'MAE')
    M[f'NineEq{tag}MaePct'] = f'{100 * (m1 / m0 - 1):.0f}'
    r = c('E9', arch, 'EqAlign trained with H2', 'EqAlign', 'top1_seen')
    if r:
        M[f'NineEq{tag}SeenDiff'] = f"{float(r['diff_B_minus_A']):.1f}"; M[f'NineEq{tag}SeenLo'] = f"{float(r['ci_lo']):.1f}"
        M[f'NineEq{tag}SeenHi'] = f"{float(r['ci_hi']):.1f}"; M[f'NineEq{tag}SeenBetter'] = r['designs_better']
    M[f'NineAug{tag}SeenBefore'] = f"{s('E1', arch, 'Augmentation', 'top1_seen'):.1f}"
    M[f'NineAug{tag}SeenAfter'] = f"{s('E9', arch, 'Augmentation, h2-trained', 'top1_seen'):.1f}"
    M[f'NineAug{tag}DHTwoAfter'] = f"{s('E9', arch, 'Augmentation, h2-trained', 'dMAE_H2'):.3f}"
import json
def S(exp, arch, config, key, suite='epfl'):
    for r in rows(exp, 'summary'):
        if r['suite'] == suite and r['arch'] == arch and r['config'] == config:
            return float(r[key + '_mean'])
def C(exp, arch, B, A, metric, suite='epfl'):
    for r in rows(exp, 'comparisons'):
        if r['suite'] == suite and r['arch'] == arch and r['B'] == B and r['A'] == A and r['metric'] == metric:
            return r
REG = ['Baseline', 'Augmentation', 'TT-sim', 'Circuit contrast', 'L1 alignment', 'L1 + TT-sim']
R = {}   # macros that replace CPU-run values (\renewcommand)
# E1a: width 256, 5 seeds
W = 'gru_d256_r2'
if rows('E1', 'summary'):
    M['WBaseRhoSyn'] = f"{S('E1', W, 'Baseline', 'rho_syn'):.2f}"; M['WBaseTop'] = f"{S('E1', W, 'Baseline', 'top1_seen'):.1f}"
    M['WOursTop'] = f"{S('E1', W, 'EqAlign lambda=0.2', 'top1_seen'):.1f}"
    d = [float(C('E1', W, 'EqAlign', a, 'top1_seen')['diff_B_minus_A']) for a in REG]
    M['WDiffMin'] = f'{min(d):.1f}'; M['WDiffMax'] = f'{max(d):.1f}'
    import re
    # EqAlign against each other regime at width 64: Welch tests over seeds, Holm-adjusted (extended version)
    _w = [float(r['p_welch_seeds_holm']) for r in rows('E1', 'comparisons') if r['suite'] == 'epfl' and r['arch'] == 'gru_d64_r2'
          and r['metric'] == 'top1_seen' and r['B'].startswith('EqAlign') and '[' not in r['B']]
    assert len(_w) == 6 and max(_w) < 1e-5, _w
    M['WelchHolmTopMax'] = '10^{-5}'
    cl = open(f'{G}/E1/claims.md').read().split('## gru_d256_r2')[1].split('##')[0]
    M['WNoise'] = re.search(r'noise floor = ([0-9.]+)x', cl).group(1)
# E3: released DeepGate2 fine-tuned on GPU, 5 seeds, the paper's configuration (5 epochs, DeepGate2's own random
# primary inputs in training, shared-input evaluation)
FC, FE = 'DG2 fine-tuned (control), 5 ep, train_pi=random', 'DG2 fine-tuned + EqAlign lambda=0.2, 5 ep, train_pi=random'
if rows('E3', 'summary'):
    n = [r for r in rows('E3', 'summary') if r['config'] == FE and r['suite'] == 'epfl'][0]['n_seeds']
    R['FtEqSeeds'] = n
    for pre, cf in (('FtCtrl', FC), ('FtEq', FE)):
        R[pre + 'TopSeen'] = f"{S('E3', 'deepgate2', cf, 'top1_seen'):.1f}"; R[pre + 'TopHo'] = f"{S('E3', 'deepgate2', cf, 'top1_ho'):.1f}"
    R['FtMaeCostPct'] = f"{100 * (S('E3', 'deepgate2', FE, 'MAE') / S('E3', 'deepgate2', FC, 'MAE') - 1):.0f}"
    for r in rows('E3', 'comparisons'):
        if r['suite'] == 'epfl' and r['metric'] == 'top1_seen' and r['B'].startswith('DG2 fine-tuned + EqAlign lambda=0.2, 5 ep, train_pi=random') \
                and '[' not in r['B'] and r['A'].startswith('DG2 fine-tuned (control), 5 ep, train_pi=random') and '[' not in r['A']:
            M['FtDiff'] = f"{float(r['diff_B_minus_A']):.1f}"; M['FtDiffLo'] = f"{float(r['ci_lo']):.1f}"
            M['FtDiffHi'] = f"{float(r['ci_hi']):.1f}"; M['FtBetter'] = r['designs_better']
# E8: positives-only cosine alignment through the same head
if rows('E8', 'summary'):
    for tag, arch, lam in (('', 'gru_d64_r2', '0.2'), ('W', W, '1')):
        M[f'Eight{tag}CosTop'] = f"{S('E8', arch, f'Cosine alignment (positives only) lambda={lam}', 'top1_seen'):.1f}"
        cl = open(f'{G}/E8/claims.md').read().split(f'## {arch}')[1]
        import re
        m = re.search(r'C7 negatives are needed.*?: \+([0-9.]+) \[([0-9.]+), ([0-9.]+)\], (\d/\d)', cl)
        M[f'Eight{tag}Diff'], M[f'Eight{tag}Better'] = m.group(1), m.group(4)
# E12: one family left out of training (3 seeds; design tests recomputed with designs lacking that family dropped)
f12 = f'{G}/E12/recomputed_design_tests.json'
if os.path.exists(f12):
    T = json.load(open(f12)); syn = [T[f'{w}_{F}'] for w in ('d64', 'd256') for F in ('S1', 'S2', 'S3', 'S4')]
    va = [t['vs_aug']['diff'] for t in syn]; cost = [-t['vs_full']['diff'] for t in syn]
    M['TwSynVsAugMin'] = f'{min(va):.1f}'; M['TwSynVsAugMax'] = f'{max(va):.1f}'; M['TwSynCostMax'] = f'{max(cost):.1f}'
    for tag, w in (('', 'd64'), ('W', 'd256')):
        t = T[f'{w}_inj']
        M[f'Tw{tag}InjOurs'] = f"{t['abs'][0]:.1f}"; M[f'Tw{tag}InjAug'] = f"{t['abs'][1]:.1f}"; M[f'Tw{tag}InjFull'] = f"{t['abs'][2]:.1f}"
        M[f'Tw{tag}InjBelowAug'] = f"{-t['vs_aug']['diff']:.1f}"; M[f'Tw{tag}InjBetter'] = f"{t['vs_aug']['n_better']}/{t['vs_aug']['n_designs']}"
        M[f'Tw{tag}OthGain'] = f"{t['others'][2]['diff']:.1f}"
# E13: large cones (test only)
if rows('E13', 'summary'):
    st = json.load(open(f'{G}/E13/large.stats.json'))
    M['LgCones'] = str(st['kept']); M['LgBuilt'] = str(st['built']); M['LgTimeout'] = str(st['dropped']['timeout'])
    M['LgDesigns'] = str(sum(v > 0 for v in st['per_design'].values()))
    M['LgGatesMin'] = f"{st['gates_min']:,}".replace(',', '{,}'); M['LgGatesMed'] = f"{st['gates_median']:,}".replace(',', '{,}')
    M['LgGatesMax'] = f"{st['gates_max']:,}".replace(',', '{,}'); M['LgNoHo'] = str(st['kept_without_heldout'])
    M['LgTimeoutMin'] = str(st['timeout_s'] // 60)
    h2 = []
    cf0 = {r['config'].split(' lambda')[0]: r['config'] for r in rows('E13', 'summary') if r['arch'] == 'gru_d64_r2'}
    for tag, arch in (('', 'gru_d64_r2'), ('W', W)):
        cf = {r['config'].split(' lambda')[0]: r['config'] for r in rows('E13', 'summary') if r['arch'] == arch}
        M[f'Lg{tag}BaseTop'] = f"{S('E13', arch, cf['Baseline'], 'top1_seen', 'large'):.1f}"
        M[f'Lg{tag}BaseRho'] = f"{S('E13', arch, cf['Baseline'], 'rho_syn', 'large'):.2f}"
        M[f'Lg{tag}OursTop'] = f"{S('E13', arch, cf['EqAlign'], 'top1_seen', 'large'):.1f}"
        M[f'Lg{tag}OtherMax'] = f"{max(S('E13', arch, cf[k], 'top1_seen', 'large') for k in REG):.1f}"
        M[f'Lg{tag}MaePct'] = f"{100 * (S('E13', arch, cf['EqAlign'], 'MAE', 'large') / S('E13', arch, cf['Baseline'], 'MAE', 'large') - 1):.0f}"
        d = [float(C('E13', arch, 'EqAlign', a, 'top1_seen', 'large')['diff_B_minus_A']) for a in REG]
        M[f'Lg{tag}DiffMin'] = f'{min(d):.1f}'
        h2 += [S('E13', arch, cf[k], 'top1_H2', 'large') for k in REG + ['EqAlign']]
    M['LgHTwoMax'] = f'{max(h2):.1f}'
    M['LgChance'] = f"{S('E13', 'gru_d64_r2', cf0['Baseline'], 'chance', 'large'):.2f}"
import glob, numpy as np
# E10: DeepGate2 retrained from its released code and data; compared with the released weights audited in the same
# GPU pipeline (E3 'DeepGate2 released', shared-input evaluation)
if rows('E10', 'summary') and rows('E3', 'summary'):
    t = [r for r in rows('E10', 'summary') if r['suite'] == 'epfl' and '[' not in r['config']][0]
    T = lambda k: float(t[k + '_mean'])
    M['TenSeeds'] = t['n_seeds']
    M['TenMAE'] = f"{T('MAE'):.3f}"; M['TenRelMAE'] = f"{S('E3', 'deepgate2', 'DeepGate2 released', 'MAE'):.3f}"
    M['TenIncons'] = f"{T('incons_rs'):.3f}"; M['TenRelIncons'] = f"{S('E3', 'deepgate2', 'DeepGate2 released', 'incons_rs'):.3f}"
    M['TenMaePct'] = f"{100 * (T('MAE') / S('E3', 'deepgate2', 'DeepGate2 released', 'MAE') - 1):.0f}"
    M['TenTop'] = f"{T('top1_seen'):.1f}"; M['TenHTwo'] = f"{T('top1_H2'):.1f}"
    M['TenRelTop'] = f"{S('E3', 'deepgate2', 'DeepGate2 released', 'top1_seen'):.1f}"; M['TenRelHTwo'] = f"{S('E3', 'deepgate2', 'DeepGate2 released', 'top1_H2'):.1f}"
    M['TenRho'] = f"{T('rho_syn'):.2f}"; M['TenRelRho'] = f"{S('E3', 'deepgate2', 'DeepGate2 released', 'rho_syn'):.2f}"
    ti = [r for r in rows('E10', 'summary') if r['suite'] == 'iscas'][0]
    M['TenIsTop'] = f"{float(ti['top1_seen_mean']):.1f}"; M['TenIsHTwo'] = f"{float(ti['top1_H2_mean']):.1f}"
    import re
    cl = open(f'{G}/E10/claims.md').read(); sc = [float(x) for x in re.findall(r'Top-1 shared ([0-9.]+)%', cl)]
    M['TenSelfMin'] = f'{min(sc):.1f}'; M['TenSelfMax'] = f'{max(sc):.1f}'
    rj = [json.load(open(f)) for f in sorted(glob.glob(f'{G}/E10/retrain_s*.json'))]
    M['TenNTrain'] = f"{rj[0]['n_train']:,}".replace(',', '{,}'); M['TenHours'] = f"{np.mean([r['sec'] for r in rj]) / 3600:.0f}"
# Scale (E1a, E1b, E2, E15): recurrent widths 64, 128, 256, 1024 and transformer widths 128, 256, 512, 5 seeds each; the baseline's
# blindness and EqAlign's lead over the baseline and augmentation at every scale (GRU: restructured pairs; transformer:
# all proven pairs, as in the paper)
_SC = [('E1', 'gru_d64_r2', 'top1_seen', 'rho_syn'), ('E1', 'gru_d128_r2', 'top1_seen', 'rho_syn'), ('E1', 'gru_d256_r2', 'top1_seen', 'rho_syn'),
       ('E1c', 'gru_d256_r4', 'top1_seen', 'rho_syn'), ('E15', 'gru_d1024_r2', 'top1_seen', 'rho_syn'), ('E2', 'tf_d128_r2', 'top1_all', 'rho_allpairs_syn'),
       ('E2', 'tf_d256_r2', 'top1_all', 'rho_allpairs_syn'), ('E15', 'tf_d512_r2', 'top1_all', 'rho_allpairs_syn')]
if all(rows(e, 'summary') for e, *_ in _SC):
    NP = json.load(open(f'{G}/n_params.json'))
    lead, gru_rho, gru_top, tf_top = [], [], [], []
    for e, arch, met, rk in _SC:
        base = [r for r in rows(e, 'summary') if r['suite'] == 'epfl' and r['arch'] == arch and r['config'] == 'Baseline'][0]
        (gru_rho if arch.startswith('gru') else []).append(float(base[rk + '_mean']))
        (gru_top if arch.startswith('gru') else tf_top).append(float(base[met + '_mean']))
        for r in rows(e, 'comparisons'):
            if r['suite'] == 'epfl' and r['arch'] == arch and r['metric'] == met and r['B'].startswith('EqAlign') \
                    and '[' not in r['B'] and r['A'] in ('Baseline', 'Augmentation'):
                lead.append(float(r['diff_B_minus_A'])); assert r['designs_better'] == '7/7', (arch, r['A'])
    M['ScFold'] = f"{NP['gru_d1024_r2'] / NP['gru_d64_r2']:.0f}"
    M['ScMinParams'] = f"{NP['gru_d64_r2'] / 1e3:.0f}k"; M['ScMaxParams'] = f"{NP['gru_d1024_r2'] / 1e6:.1f}M"
    M['ScBaseRhoMin'] = f'{min(gru_rho):.2f}'; M['ScBaseRhoMax'] = f'{max(gru_rho):.2f}'
    M['ScBaseTopFirst'] = f'{gru_top[0]:.1f}'; M['ScBaseTopLast'] = f'{gru_top[-1]:.1f}'
    M['ScTfTopMin'] = f'{min(tf_top):.1f}'; M['ScTfTopMax'] = f'{max(tf_top):.1f}'
    M['ScLeadMin'] = f'{min(lead):.1f}'; M['ScLeadMax'] = f'{max(lead):.1f}'
    _l128 = [(float(r['diff_B_minus_A']), r['designs_better']) for r in rows('E1', 'comparisons') if r['suite'] == 'epfl'
             and r['arch'] == 'gru_d128_r2' and r['metric'] == 'top1_seen' and r['B'].startswith('EqAlign') and '[' not in r['B']]
    assert len(_l128) == 6 and all(b == '7/7' for _, b in _l128), _l128   # EqAlign against each of the six other regimes
    M['ScOneTwoEightMin'] = f'{min(d for d, _ in _l128):.1f}'; M['ScOneTwoEightMax'] = f'{max(d for d, _ in _l128):.1f}'
    _ld = [(float(r['diff_B_minus_A']), r['designs_better']) for r in rows('E1c', 'comparisons') if r['suite'] == 'epfl'
           and r['arch'] == 'gru_d256_r4' and r['metric'] == 'top1_seen' and r['B'].startswith('EqAlign') and '[' not in r['B']]
    assert len(_ld) == 6 and all(b == '7/7' for _, b in _ld), _ld   # 4 rounds: EqAlign against each of the six other regimes
    M['ScDeepLeadMin'] = f'{min(d for d, _ in _ld):.1f}'; M['ScDeepLeadMax'] = f'{max(d for d, _ in _ld):.1f}'
    _bd = [r for r in rows('E1c', 'summary') if r['suite'] == 'epfl' and r['config'] == 'Baseline'][0]
    M['ScDeepBaseRho'] = f"{float(_bd['rho_syn_mean']):.2f}"; M['ScDeepBaseTop'] = f"{float(_bd['top1_seen_mean']):.1f}"
    _is = {r['config']: r for r in rows('E15', 'summary') if r['suite'] == 'iscas' and r['arch'] == 'gru_d1024_r2'}
    M['ScIsBase'] = f"{float(_is['Baseline']['top1_seen_mean']):.1f}"
    M['ScIsOurs'] = f"{float([v for c, v in _is.items() if c.startswith('EqAlign')][0]['top1_seen_mean']):.1f}"
# E8: the projection head's contribution (EqAlign against EqAlign without head, width 64, 5 seeds)
if rows('E8', 'summary'):
    import re
    _c8 = open(f'{G}/E8/claims.md').read().split('## gru_d64_r2')[1].split('\n## ')[0]
    _m = re.search(r'EqAlign minus EqAlign \(no head\) lambda=[0-9.]+: \+([0-9.]+) \[([0-9.]+), ([0-9.]+)\], (\d/\d)', _c8)
    M['NoHeadDiff'], M['NoHeadLo'], M['NoHeadHi'], M['NoHeadBetter'] = _m.group(1), _m.group(2), _m.group(3), _m.group(4)
with open('paper/numbers.tex', 'a') as fh:
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v.replace("-", chr(92) + "ensuremath{-}")}}}\n')
    for k, v in R.items():
        fh.write(f'\\renewcommand{{\\{k}}}{{{v.replace("-", chr(92) + "ensuremath{-}")}}}\n')
print(R)
print(M)
