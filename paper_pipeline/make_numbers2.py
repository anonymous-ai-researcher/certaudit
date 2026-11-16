"""Revised number generation: every number quoted in the paper is written to paper/numbers.tex here."""
import re, json, pickle, glob, os, numpy as np
from analysis import load, summary, group_summary, group_metrics, adv_matched, design_test, design_means, SYN, INJ
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # paths below are repo-relative

LAM = os.environ.get('LAMSTAR', '0.2'); TTL = os.environ.get('TTLAM', '0.1'); CL = os.environ.get('CIRCLAM', '0.2')
PREF = {'Base': 'base', 'Aug': 'aug', 'TT': f'ttsim_ph_l{TTL}', 'Circ': f'circ_ph_l{CL}', 'Ours': f'eqalign_ph_l{LAM}'}
M = {}
D = pickle.load(open('data.pkl', 'rb')); A = pickle.load(open('adv.pkl', 'rb')); H = pickle.load(open('heldout.pkl', 'rb'))
st = json.load(open('match_stats.json'))
M['NTrainConesAll'] = f"{len(D['train']):,}"; M['NValCones'] = '275'; M['NTrainCones'] = f"{len(D['train']) - 275:,}"
M['NTestCones'] = f"{len(D['test']):,}"
nfix = sum(len(r) for r in D['tvars']); nadv = sum(len(a) for a in A); nho = 2 * sum(h is not None for h in H)
M['NFixedVar'] = f'{nfix:,}'; M['NAdvVar'] = f'{nadv:,}'; M['NHoVar'] = f'{nho:,}'; M['NTotalVar'] = f'{nfix + nadv + nho:,}'
M['NHoCones'] = str(sum(h is not None for h in H)); M['NHoFail'] = str(sum(h is None for h in H))
npairs = sum(len(v['I']) for r in D['tvars'] for v in r.values()) + sum(len(v['I']) for a in A for v in a) + \
    sum(len(h[k]['I']) for h in H if h for k in ('H1', 'H2'))
M['NTestPairsK'] = f'{npairs / 1000:.0f}k'
t = st['test']
M['SigCandTest'] = f"{t['sig_nodes']:,}"; M['SigFalseTest'] = f"{t['sig_nodes'] - t['proven']:,}"
M['SigFalsePct'] = f"{100 * (t['sig_nodes'] - t['proven']) / t['sig_nodes']:.1f}"
M['SigCallsTest'] = f"{t['proven'] + t['refuted_calls']:,}"; M['SigRefCallsPct'] = f"{100 * t['refuted_calls'] / (t['proven'] + t['refuted_calls']):.0f}"
fm = json.load(open('falsematch.json'))
M['SigFalseSynPct'] = f"{fm['syn'][2]:.1f}"; M['SigFalseInjPct'] = f"{fm['inj'][2]:.1f}"
M['SigNearConstSynPct'] = f"{fm['syn'][3]}"; M['SigNearConstInjPct'] = f"{fm['inj'][3]}"
M['LamStar'] = LAM; M['TTLamStar'] = TTL; M['CircLamStar'] = CL
_g = np.array([len(d['fanins']) for d in D['test']]); _i = np.array([d['n_pi'] for d in D['test']])
M['GatesMin'], M['GatesMed'], M['GatesMax'] = str(_g.min()), str(int(np.median(_g))), str(_g.max())
M['InMed'], M['InMax'] = str(int(np.median(_i))), str(_i.max())
# share of complemented proven pairs among the TRAINING pairs EqAlign draws from (the 2,475 training cones of the
# fixed split; the 275 validation cones are excluded, as in train.py)
_tr = np.sort(np.random.default_rng(12345).permutation(len(D['train']))[275:])
_c = np.concatenate([D['pairs'][i]['P'] for i in _tr])
M['ComplPct'] = f'{100 * (_c == 1).mean():.0f}'

import re as _re
SEEDS = lambda p: sorted(x for x in glob.glob(f'raw_{p}_s*.pkl') if _re.fullmatch(rf'raw_{_re.escape(p)}_s\d+\.pkl', x))
RAW = {k: [load(x[4:-4]) for x in SEEDS(p)] for k, p in PREF.items()}
RAW = {k: v for k, v in RAW.items() if v}


def per_seed(k, f):
    v = np.array([f(R) for R in RAW[k]]); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


G = lambda kinds=None, ho=None, key=None: (lambda R: group_summary(R, kinds=kinds, heldout=ho)[0][key])
HOAVG = lambda key: (lambda R: np.mean([group_summary(R, heldout=h)[0][key] for h in ('H1', 'H2')]))
specs = {
    'MAE': (lambda R: np.mean([r['mae_orig'] for r in R]), '{:.4f}', 1),
    'DSyn': (G(SYN, key='mae_m_gap'), '{:.4f}', 1), 'DInj': (G(INJ, key='mae_m_gap'), '{:.4f}', 1),
    'DHo': (HOAVG('mae_m_gap'), '{:.4f}', 1),
    'RhoSyn': (G(SYN, key='rho'), '{:.2f}', 1), 'RhoInj': (G(INJ, key='rho'), '{:.2f}', 1), 'RhoHo': (HOAVG('rho'), '{:.2f}', 1),
    'RhoAll': (G(SYN + INJ, key='rho'), '{:.2f}', 1),
    'IncRS': (G(SYN + INJ, key='incons_rs'), '{:.4f}', 1), 'ErrRS': (G(SYN + INJ, key='err_rs'), '{:.4f}', 1),
    'IncRSSyn': (G(SYN, key='incons_rs'), '{:.4f}', 1),
    'Worst': (lambda R: adv_matched(R)['worst'], '{:.4f}', 1), 'WorstOrig': (lambda R: adv_matched(R)['orig'], '{:.4f}', 1),
    'WorstRatio': (lambda R: adv_matched(R)['ratio'], '{:.2f}', 1),
    'TopRS': (G(SYN + INJ, key='top1_rs'), '{:.1f}', 100), 'TopSyn': (G(SYN, key='top1_rs'), '{:.1f}', 100),
    'TopInj': (G(INJ, key='top1_rs'), '{:.1f}', 100), 'TopHo': (HOAVG('top1_rs'), '{:.1f}', 100),
    'AUROC': (G(SYN + INJ, key='auroc'), '{:.3f}', 1), 'AUCHard': (G(SYN + INJ, key='auroc_hard'), '{:.3f}', 1),
}
SUM = {}
for k in RAW:
    for nm, (f, fmt, sc) in specs.items():
        mu, sd = per_seed(k, f); SUM[(k, nm)] = mu
        M[f'{k}{nm}'] = fmt.format(mu * sc); M[f'{k}{nm}Sd'] = fmt.format(sd * sc)
    M[f'{k}NSeeds'] = str(len(RAW[k]))
M['Chance'] = f"{100 * group_summary(RAW['Base'][0], kinds=SYN + INJ)[0]['chance']:.1f}"
# Depth of matched gates before and after a rewrite: synthesis makes them shallower and injection
# deeper, which moves every model's error alike (why synthesis dMAE is the same for all regimes).
def _depth(kinds):
    o, v = [], []
    for d, row in zip(D['test'], D['tvars']):
        for k in kinds:
            r = row.get(str(k))
            if r is not None and len(r['I']):
                o.append(np.mean(d['lev'][r['I']])); v.append(np.mean(r['var']['lev'][r['J']]))
    return np.mean(o), np.mean(v)
_so, _sv = _depth(SYN); _io, _iv = _depth(INJ)
M['DepthSynOrig'], M['DepthSynVar'] = f'{_so:.1f}', f'{_sv:.1f}'
M['DepthInjOrig'], M['DepthInjVar'] = f'{_io:.1f}', f'{_iv:.1f}'
# DeepGate2-faithful TT-sim: the same objective with DeepGate2's own pair filters applied.
# Evaluated without the adversarial sweep, so it is kept out of the main spec loop.
TTF = [load(x[4:-4]) for x in sorted(glob.glob(f'raw_ttsimf_ph_l{TTL}_s*.pkl'))]
if TTF:
    _f = lambda key, kinds=SYN + INJ, ho=None: np.mean([group_summary(R, kinds=kinds, heldout=ho)[0][key] for R in TTF])
    M['TTFNSeeds'] = str(len(TTF))
    M['TTFMAE'] = f"{np.mean([np.mean([r['mae_orig'] for r in R]) for R in TTF]):.4f}"
    M['TTFRhoAll'] = f"{_f('rho'):.2f}"
    M['TTFRhoSyn'] = f"{_f('rho', SYN):.2f}"; M['TTFRhoInj'] = f"{_f('rho', INJ):.2f}"
    M['TTFDInj'] = f"{_f('mae_m_gap', INJ):.4f}"
    M['TTFTopRS'] = f"{100 * _f('top1_rs'):.1f}"
    M['TTFTopRSSd'] = f"{100 * np.std([group_summary(R, kinds=SYN + INJ)[0]['top1_rs'] for R in TTF], ddof=1):.1f}"
    M['TTFTopHo'] = f"{100 * np.mean([np.mean([group_summary(R, heldout=h)[0]['top1_rs'] for h in ('H1', 'H2')]) for R in TTF]):.1f}"
    M['TTFAUCHard'] = f"{_f('auroc_hard'):.3f}"
pct = lambda a, b: f'{100 * (a / b - 1):.0f}'
M['OursMAECostPct'] = pct(SUM[('Ours', 'MAE')], SUM[('Base', 'MAE')]); M['OursMAECostAugPct'] = pct(SUM[('Ours', 'MAE')], SUM[('Aug', 'MAE')])
M['BaseIncVsErrPct'] = f"{100 * SUM[('Base', 'RhoAll')]:.0f}"
M['AugIncRedPct'] = f"{100 * (1 - SUM[('Aug', 'IncRS')] / SUM[('Base', 'IncRS')]):.0f}"
M['OursIncRedPct'] = f"{100 * (1 - SUM[('Ours', 'IncRS')] / SUM[('Base', 'IncRS')]):.0f}"
if 'TT' in RAW:
    M['TTMAECostPct'] = pct(SUM[('TT', 'MAE')], SUM[('Base', 'MAE')])

# design-level tests (seeds pooled; cluster = source design)
def dt(a, b, getter, name, scale=1.0):
    r = design_test(RAW[a], RAW[b], getter)
    M[f'DT{name}Diff'] = f"{scale * r['diff']:.{1 if scale == 100 else 4}f}"
    M[f'DT{name}Lo'] = f"{scale * r['lo']:.{1 if scale == 100 else 4}f}"; M[f'DT{name}Hi'] = f"{scale * r['hi']:.{1 if scale == 100 else 4}f}"
    M[f'DT{name}Npos'] = f"{r['n_pos']}"; M[f'DT{name}N'] = f"{r['n']}"; M[f'DT{name}P'] = f"{r['p_sign']:.3f}"
    return r
gm = lambda key, kinds=SYN + INJ: (lambda R: group_metrics(R, kinds=kinds)[key])
dt('Base', 'Ours', gm('top1_rs'), 'TopOB', 100)
dt('Aug', 'Ours', gm('top1_rs'), 'TopOA', 100)
if 'TT' in RAW:
    dt('TT', 'Ours', gm('top1_rs'), 'TopOT', 100)
if 'Circ' in RAW:
    dt('Circ', 'Ours', gm('top1_rs'), 'TopOC', 100)
# RQ2 states one sign-test p for the four comparisons and that each holds on all 7 designs with its interval above 0
_rq2 = [M[f'DT{n}P'] for n in ('TopOB', 'TopOA', 'TopOT', 'TopOC')]
assert len(set(_rq2)) == 1 and all(M[f'DT{n}Npos'] == M[f'DT{n}N'] == '7' for n in ('TopOB', 'TopOA', 'TopOT', 'TopOC')), 'RQ2 sign tests'
assert all(float(M[f'DT{n}Lo']) > 0 for n in ('TopOB', 'TopOA', 'TopOT', 'TopOC')), 'RQ2 intervals'
dt('Base', 'Ours', lambda R: np.array([r['mae_orig'] for r in R]), 'MaeOB')
dt('Base', 'Aug', gm('incons_rs'), 'IncAB')
dt('Base', 'Aug', lambda R: np.array([r['mae_orig'] for r in R]), 'MaeAB')

# noise floor
if os.path.exists('noise_floor.json'):
    nf = json.load(open('noise_floor.json'))
    M['NoiseBase'] = f"{nf['base']:.4f}"
    M['NoiseRatioBase'] = f"{nf['base'] / SUM[('Base', 'ErrRS')]:.2f}"
    M['IncOverNoiseBase'] = f"{SUM[('Base', 'IncRS')] / nf['base']:.1f}"
    M['IncSynOverNoiseBase'] = f"{SUM[('Base', 'IncRSSyn')] / nf['base']:.1f}"
# recall
if os.path.exists('recall_at_k.json'):
    rk = json.load(open('recall_at_k.json'))
    for k, m in [('Base', 'base'), ('Aug', 'aug'), ('TT', 'tt'), ('Ours', 'ours')]:
        if m in rk and isinstance(rk[m], dict):
            M[f'Rec{k}One'] = f"{rk[m]['mean'][0]:.0f}"; M[f'Rec{k}Five'] = f"{rk[m]['mean'][3]:.0f}"
# lambda observation (seed 0)
if os.path.exists('raw_eqalign_ph_l0.05_s0.pkl'):
  s005 = load('eqalign_ph_l0.05_s0'); M['LowLamMAE'] = f"{np.mean([r['mae_orig'] for r in s005]):.4f}"
  M['LowLamTop'] = f"{100 * group_summary(s005, kinds=SYN + INJ)[0]['top1_rs']:.1f}"
M['BaseSzeroMAE'] = f"{np.mean([r['mae_orig'] for r in RAW['Base'][0]]):.4f}"

# SAT sweeping
def sweep_stats(files):
    """Same aggregation as Fig. 4 (sweepagg.reduction): pooled per seed, mean over seeds."""
    from sweepagg import reduction
    rows = [x for f in files for x in json.load(open(f))]
    res = {}
    for bits in sorted(set(x['bits'] for x in rows)):
        for pol in ('random', 'level', 'emb', 'oracle'):
            res[(bits, pol)] = dict(
                red_w=reduction(files, pol, bits, 'wasted')[0],
                red_t=reduction(files, pol, bits, 'time')[0],
                red_te=reduction(files, pol, bits, 'time', add_emb=True)[0],
                topo_w=float(np.mean([x['wasted'] for x in rows if x['bits'] == bits and x['policy'] == 'topo'])),
                topo_t=1000 * float(np.mean([x['time'] for x in rows if x['bits'] == bits and x['policy'] == 'topo'])),
                emb_ms=1000 * float(np.mean([x['emb_time'] for x in rows])))
    return res
for k, p in PREF.items():
    fs = sorted(glob.glob(f'sweep2_v_{p}_s*.json'))
    if not fs:
        continue
    S = sweep_stats(fs)
    M[f'Sw{k}EmbSlowEight'] = f"{-S[(8, 'emb')]['red_te']:.0f}"
    for bits in (8, 64):
        M[f'Sw{k}Emb{"Eight" if bits == 8 else "SixtyFour"}'] = f"{S[(bits, 'emb')]['red_w']:.0f}"
        M[f'Sw{k}EmbTime{"Eight" if bits == 8 else "SixtyFour"}'] = f"{S[(bits, 'emb')]['red_te']:.0f}"
        M[f'Sw{k}EmbSatTime{"Eight" if bits == 8 else "SixtyFour"}'] = f"{S[(bits, 'emb')]['red_t']:.0f}"
        if k == 'Ours':
            for pol, nm in (('level', 'Level'), ('oracle', 'True'), ('random', 'Random')):
                M[f'Sw{nm}{"Eight" if bits == 8 else "SixtyFour"}'] = f"{S[(bits, pol)]['red_w']:.0f}"
            M[f'SwTopoCalls{"Eight" if bits == 8 else "SixtyFour"}'] = f"{S[(bits, 'emb')]['topo_w']:.0f}"
            M[f'SwTopoMs{"Eight" if bits == 8 else "SixtyFour"}'] = f"{S[(bits, 'emb')]['topo_t']:.0f}"
    if k == 'Ours':
        M['SwEmbMs'] = f"{S[(8, 'emb')]['emb_ms']:.0f}"
        _cf = sorted(glob.glob(f'cpu_sweep2_v_{p}_s*.json'))
        if _cf:                                              # timings of the CPU run (inference on one dedicated core)
            Sc = sweep_stats(_cf)
            M['SwCpuSlowEight'] = f"{-Sc[(8, 'emb')]['red_te']:.0f}"; M['SwCpuEmbMs'] = f"{Sc[(8, 'emb')]['emb_ms']:.0f}"
            M['SwCpuSatTimeEight'] = f"{Sc[(8, 'emb')]['red_t']:.0f}"
# simulation-signature retrieval reference
if os.path.exists('sigref.json'):
    sr = json.load(open('sigref.json'))
    M['SigRefEight'] = f"{100 * sr['8']:.0f}"; M['SigRefSixtyFour'] = f"{100 * sr['64']:.0f}"; M['SigRefK'] = f"{100 * sr['1024']:.0f}"
# second suite (ISCAS) and bidirectional encoder
def load_is(t):
    return pickle.load(open(f'rawis_{t}.pkl', 'rb'))
IS = {k: [load_is(x[6:-4]) for x in sorted(glob.glob(f'rawis_{p}_s*.pkl'))] for k, p in PREF.items()}
IS = {k: v for k, v in IS.items() if v}
if IS:
    M['ISCones'] = str(len(next(iter(IS.values()))[0])); M['ISDesigns'] = str(len(set(r['src'] for r in next(iter(IS.values()))[0])))
    for k, Rs in IS.items():
        f = lambda key, kinds=None, ho=None: np.mean([group_summary(R, kinds=kinds, heldout=ho)[0][key] for R in Rs])
        hoavg = lambda key: np.mean([np.mean([group_summary(R, heldout=h)[0][key] for h in ('H1', 'H2')]) for R in Rs])
        M[f'IS{k}RhoSyn'] = f"{f('rho', SYN):.2f}"; M[f'IS{k}RhoInj'] = f"{f('rho', INJ):.2f}"; M[f'IS{k}RhoHo'] = f"{hoavg('rho'):.2f}"
        M[f'IS{k}DInj'] = f"{f('mae_m_gap', INJ):.4f}"; M[f'IS{k}DHo'] = f"{hoavg('mae_m_gap'):.4f}"
        M[f'IS{k}TopRS'] = f"{100 * f('top1_rs', SYN + INJ):.1f}"; M[f'IS{k}TopHo'] = f"{100 * hoavg('top1_rs'):.1f}"
        M[f'IS{k}MAE'] = f"{np.mean([np.mean([r['mae_orig'] for r in R]) for R in Rs]):.4f}"
# encoders that read beyond the fanin cone (bidirectional, transformer): metrics over all proven pairs
def rho_all(R, kinds):
    c = group_metrics(R, kinds=kinds); return np.nanmean(c['incons']) / np.nanmean(c['mae_m_orig'])
for _pre, _tag, _lam in (('Bi', 'bi', os.environ.get('BILAM', LAM)), ('Tf', 'tf', os.environ.get('TFLAM', LAM))):
    BIP = {'Base': f'{_tag}_base', 'Aug': f'{_tag}_aug', 'Ours': f'{_tag}_eqalign_ph_l{_lam}'}
    BI = {k: [load(x[4:-4]) for x in sorted(glob.glob(f'raw_{p}_s*.pkl')) if re.fullmatch(rf'raw_{re.escape(p)}_s\d+\.pkl', x)] for k, p in BIP.items()}
    BI = {k: v for k, v in BI.items() if v}
    if BI:
        M[f'{_pre}NSeeds'] = str(len(BI['Base']))
    for k, Rs in BI.items():
        M[f'{_pre}{k}RhoSyn'] = f"{np.mean([rho_all(R, SYN) for R in Rs]):.2f}"; M[f'{_pre}{k}RhoInj'] = f"{np.mean([rho_all(R, INJ) for R in Rs]):.2f}"
        M[f'{_pre}{k}RhoHo'] = f"{np.mean([np.mean([np.nanmean(group_metrics(R, heldout=h)['incons']) / np.nanmean(group_metrics(R, heldout=h)['mae_m_orig']) for h in ('H1', 'H2')]) for R in Rs]):.2f}"
        M[f'{_pre}{k}DInj'] = f"{np.mean([group_summary(R, kinds=INJ)[0]['mae_m_gap'] for R in Rs]):.4f}"
        M[f'{_pre}{k}DHo'] = f"{np.mean([np.mean([group_summary(R, heldout=h)[0]['mae_m_gap'] for h in ('H1', 'H2')]) for R in Rs]):.4f}"
        M[f'{_pre}{k}TopRS'] = f"{100 * np.mean([group_summary(R, kinds=SYN + INJ)[0]['top1'] for R in Rs]):.1f}"
        M[f'{_pre}{k}TopHo'] = f"{100 * np.mean([np.mean([group_summary(R, heldout=h)[0]['top1'] for h in ('H1', 'H2')]) for R in Rs]):.1f}"
        M[f'{_pre}{k}MAE'] = f"{np.mean([np.mean([r['mae_orig'] for r in R]) for R in Rs]):.4f}"
# Experiments A (pair source) and B (frozen-encoder probe)
def _top(prefix):
    Rs = [load(x[4:-4]) for x in SEEDS(prefix)]
    return np.array([group_summary(R, kinds=SYN + INJ)[0]['top1_rs'] for R in Rs]) if Rs else None
_AB = {'ProbeBase': f'eqalign_probebase_ph_l{LAM}', 'ProbeAug': f'eqalign_probeaug_ph_l{LAM}',
       'SigSixtyFour': f'eqalign_sig64_ph_l{LAM}', 'SigFourK': f'eqalign_sig4096_ph_l{LAM}'}
_tv = {k: _top(p) for k, p in _AB.items()}
for k, v in _tv.items():
    if v is not None:
        M[f'{k}Top'] = f"{100 * v.mean():.1f}"; M[f'{k}TopSd'] = f"{100 * v.std(ddof=1):.1f}"
if _tv['ProbeBase'] is not None:
    _b = np.mean([group_summary(R, kinds=SYN + INJ)[0]['top1_rs'] for R in RAW['Base']])
    _o = np.mean([group_summary(R, kinds=SYN + INJ)[0]['top1_rs'] for R in RAW['Ours']])
    M['EncShare'] = f"{100 * (_o - _tv['ProbeBase'].mean()) / (_o - _b):.0f}"
# In-pipeline independent-error null (rho_null2.py): gap closed g = 1 - rho / rho_perp
if os.path.exists('rho_null2.json'):
    _rn = json.load(open('rho_null2.json'))
    for k, p in PREF.items():
        tags = [t for t in _rn if t.rsplit('_s', 1)[0] == p]
        if len(tags) < 2: continue
        for fam, F in (('syn', 'Syn'), ('inj', 'Inj')):
            M[f'{k}Gap{F}'] = f"{100 * np.mean([_rn[t][fam]['gap'] for t in tags]):.0f}"
            M[f'{k}RhoPerp{F}'] = f"{np.mean([_rn[t][fam]['rho_perp'] for t in tags]):.2f}"
            M[f'{k}RhoPool{F}'] = f"{np.mean([_rn[t][fam]['rho'] for t in tags]):.2f}"
# Confirmatory design (EPFL hyp, unused during development; hyp_build.py, DATA_TAG=hyp)
if glob.glob('hyp_raw/raw_base_s0_hyp.pkl'):
    # the confirmatory run keeps its own models (CPU environment, 3 seeds, TT-sim weight selected there): HYP_TTLAM
    _HP = dict(PREF, TT=f"ttsim_ph_l{os.environ.get('HYP_TTLAM', TTL)}")
    HY = {k: [pickle.load(open(f'hyp_raw/raw_{p}_s{i}_hyp.pkl', 'rb')) for i in range(3)] for k, p in _HP.items()}
    M['HypCones'] = str(len(HY['Base'][0]))
    M['HypBaseRhoSyn'] = f"{np.mean([group_summary(R, kinds=SYN)[0]['rho'] for R in HY['Base']]):.2f}"
    _ht = {k: np.array([group_summary(R, kinds=SYN + INJ)[0]['top1_rs'] for R in Rs]) for k, Rs in HY.items()}
    M['HypBaseTop'] = f"{100 * _ht['Base'].mean():.1f}"; M['HypOursTop'] = f"{100 * _ht['Ours'].mean():.1f}"
    M['HypBestOtherTop'] = f"{100 * max(v.mean() for k, v in _ht.items() if k != 'Ours'):.1f}"
    M['HypOursTopMin'] = f"{100 * _ht['Ours'].min():.1f}"
    M['HypOtherTopMax'] = f"{100 * max(v.max() for k, v in _ht.items() if k != 'Ours'):.1f}"
    _h2 = [np.mean([group_summary(R, heldout='H2')[0]['mae_m_gap'] for R in Rs]) for Rs in HY.values()]
    M['HypHTwoMin'] = f"{min(_h2):.3f}"; M['HypHTwoMax'] = f"{max(_h2):.3f}"
    # which EPFL findings replicate on hyp (the main text names the ones that do; the extended version the ones that do not)
    _hm = lambda k, key, kinds: np.mean([group_summary(R, kinds=kinds)[0][key] for R in HY[k]])
    M['HypBaseRhoInj'] = f"{_hm('Base', 'rho', INJ):.2f}"; M['HypAugRhoInj'] = f"{_hm('Aug', 'rho', INJ):.2f}"
    assert _hm('Aug', 'rho', INJ) < _hm('Base', 'rho', INJ)
    _ha = {k: _hm(k, 'auroc_hard', SYN + INJ) for k in HY}
    M['HypOursAUC'] = f"{_ha['Ours']:.3f}"; M['HypBestOtherAUC'] = f"{max(v for k, v in _ha.items() if k != 'Ours'):.3f}"
    _hmae = {k: np.mean([np.mean([r['mae_orig'] for r in R]) for R in HY[k]]) for k in HY}
    assert min(_hmae, key=_hmae.get) == 'Ours' and _ha['Ours'] < max(_ha.values())   # the two findings that do not replicate
    M['HypOursMAE'] = f"{np.mean([np.mean([r['mae_orig'] for r in R]) for R in HY['Ours']]):.4f}"
    M['HypBaseMAE'] = f"{np.mean([np.mean([r['mae_orig'] for r in R]) for R in HY['Base']]):.4f}"
# hyp with the GPU models (supplementary replication; the confirmatory evaluation above is the CPU run's, made once)
if os.path.exists('hyp_gpu.pkl'):
    HG = pickle.load(open('hyp_gpu.pkl', 'rb'))
    _hg = {k: np.array([group_summary(R, kinds=SYN + INJ)[0]['top1_rs'] for R in v]) for k, v in HG.items()}
    M['HypGpuSeeds'] = str(len(HG['Ours'])); M['HypGpuOursTop'] = f"{100 * _hg['Ours'].mean():.1f}"
    M['HypGpuBaseTop'] = f"{100 * _hg['Base'].mean():.1f}"
    M['HypGpuBestOtherTop'] = f"{100 * max(v.mean() for k, v in _hg.items() if k != 'Ours'):.1f}"
    M['HypGpuOursTopMin'] = f"{100 * _hg['Ours'].min():.1f}"
    M['HypGpuOtherTopMax'] = f"{100 * max(v.max() for k, v in _hg.items() if k != 'Ours'):.1f}"
    _g2 = [np.mean([group_summary(R, heldout='H2')[0]['mae_m_gap'] for R in v]) for v in HG.values()]
    M['HypGpuHTwoMin'] = f"{min(_g2):.3f}"
M['BaseWorstMed'] = f"{np.median(np.concatenate([adv_matched(R)['worst_c'] / adv_matched(R)['orig_c'] for R in RAW['Base']])):.2f}"
# Retrieval on the encoder state h instead of the projection z (EqAlign, 3 seeds; hret_raw/)
if glob.glob('hret_raw/raw_eqalign_ph_l0.2_s0_hret.pkl'):
    _HR = [pickle.load(open(f, 'rb')) for f in sorted(glob.glob(f'hret_raw/raw_eqalign_ph_l{LAM}_s*_hret.pkl'))]
    M['HretTop'] = f"{100 * np.mean([group_summary(R, kinds=SYN + INJ)[0]['top1_rs'] for R in _HR]):.1f}"
    M['HretTopHo'] = f"{100 * np.mean([np.mean([group_summary(R, heldout=h)[0]['top1_rs'] for h in ('H1', 'H2')]) for R in _HR]):.1f}"
# MixGate-style L1 alignment (positives only, on h) and L1 + TT-sim; weights selected on validation
for _k, _p in (('Lone', f"l1align_l{os.environ.get('L1LAM', '0.003')}"), ('Lonett', f"l1tt_l{os.environ.get('L1TTLAM', '0.003')}")):
    _R = [load(x[4:-4]) for x in SEEDS(_p)]
    if len(_R) < 3: continue
    _t = np.array([group_summary(R, kinds=SYN + INJ)[0]['top1_rs'] for R in _R])
    M[f'{_k}Top'] = f"{100 * _t.mean():.1f}"; M[f'{_k}TopSd'] = f"{100 * _t.std(ddof=1):.1f}"
    M[f'{_k}TopHo'] = f"{100 * np.mean([np.mean([group_summary(R, heldout=h)[0]['top1_rs'] for h in ('H1', 'H2')]) for R in _R]):.1f}"
    M[f'{_k}MAE'] = f"{np.mean([np.mean([r['mae_orig'] for r in R]) for R in _R]):.4f}"
    M[f'{_k}RhoInj'] = f"{np.mean([group_summary(R, kinds=INJ)[0]['rho'] for R in _R]):.2f}"
    M[f'{_k}RhoSyn'] = f"{np.mean([group_summary(R, kinds=SYN)[0]['rho'] for R in _R]):.2f}"
    M[f'{_k}Inc'] = f"{np.mean([group_summary(R, kinds=SYN + INJ)[0]['incons_rs'] for R in _R]):.4f}"
    M[f'{_k}Lam'] = _p.rsplit('_l', 1)[1]
with open('paper/numbers.tex', 'w') as fh:
    for k, v in M.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
json.dump(M, open('numbers.json', 'w'), indent=1)

# per-design table
designs = sorted(set(r['src'] for r in RAW['Base'][0]))
L = ['\\begin{tabular}{l r cc cc}', '\\toprule', ' & & \\multicolumn{2}{c}{MAE (original)} & \\multicolumn{2}{c}{Top-1 rs.\\ (\\%)} \\\\',
     '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}', 'Design & cones & Base & \\ours & Base & \\ours \\\\', '\\midrule']
for s in designs:
    nc = sum(r['src'] == s for r in RAW['Base'][0])
    mb = np.mean([design_means(R, np.array([r['mae_orig'] for r in R]))[s] for R in RAW['Base']])
    mo = np.mean([design_means(R, np.array([r['mae_orig'] for r in R]))[s] for R in RAW['Ours']])
    tb = np.mean([design_means(R, group_metrics(R, kinds=SYN + INJ)['top1_rs'])[s] for R in RAW['Base']])
    to = np.mean([design_means(R, group_metrics(R, kinds=SYN + INJ)['top1_rs'])[s] for R in RAW['Ours']])
    L.append(f"{s.split('/')[1].replace('_', '\\_')} & {nc} & {mb:.4f} & {mo:.4f} & {100 * tb:.1f} & {100 * to:.1f} \\\\")
L += ['\\bottomrule', '\\end{tabular}']
open('paper/tab_design.tex', 'w').write('\n'.join(L) + '\n')
print(json.dumps(M, indent=1))
