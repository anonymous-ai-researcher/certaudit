"""Job lists for every experiment and phase (nothing here trains). Writes jobs/<exp>_<phase>.jsonl for
tools/run_queue.py. Phases after a selection read runs/<exp>/selection.json (tools/select.py, validation only).
Every grid, seed list and rule below is fixed BEFORE any result is seen. run_all.sh runs everything in the
priority order of README section 5.

  smoke                     1-epoch end-to-end test of every job type
  E0                        the paper's exact setting on GPU (7 regimes x 3 seeds) + noise floor
  E3  grid | final          DeepGate2: released checkpoint; fine-tuning (control / EqAlign), PI protocols
  E1a lr | grid | final     model scale, GRU encoder, widths 64 and 256 (priority)
  E8  grid | final          mechanism ablations at widths 64 and 256 (needs E1a selection and final)
  E9  final                 H2 positive control: train with XOR re-encoded variants (needs E1a selection)
  E1b lr | grid | final     model scale, widths 128 and 512, and 256 with 4 rounds
  E4  lr | grid | final     training-data scale: 1, 2, 4 certified variants per cone
  E2  lr | grid | final     architectures: bidirectional (DeepGate-like) and transformer encoders
  E12 final                 leave-one-family-out: hold out S1, S2, S3, S4 or injection in training (needs E1a selection
                            and data/pairs_extra.pkl); tests whether repairs transfer to an unseen synthesis family
  E13 data | eval           large-circuit test suite (test only, E1a final models; CPU-heavy data step)
  E10                       DeepGate2 retrained from scratch with its official code and data (3 seeds), audited
  E5                        SAT-sweeping probe with the E1a/E1b final models (CPU-heavy, optional)
  E15 lr | grid | final     larger encoders (v4.3): GRU width 1024 and transformer width 512; baseline, augmentation and
                            EqAlign only (EqAlign weight grid), lr grid shifted down for the larger widths, 5 seeds
  E14 (supplement, v4.2)    (a) EqAlign trained on simulation-signature pairs (64 and 4,096 bits) instead of SAT-proven
                            ones, E1a weights and lr, 5 seeds; (b) every E1a final model evaluated once on the
                            confirmatory design hyp; (c) recall@k and the Fig. 1 example from the E1a final models;
                            (d) test-set evaluation of the E1a weight-grid runs (seed 0) for the ablation table
"""
import glob, json, os, sys

SEEDS = [0, 1, 2, 3, 4]
LRS = [2e-3, 1e-3, 5e-4]                    # chosen per architecture on the BASELINE's validation MAE
GRIDS = {
    'base': [None], 'aug': [None],
    'ttsim_ph': [0.03, 0.1, 0.5, 1.0, 2.0],
    'circ_ph': [0.05, 0.2, 0.5, 1.0],
    'l1align': [0.0003, 0.001, 0.003, 0.01, 0.03],
    'l1tt': [0.001, 0.003, 0.01],
    'eqalign_ph': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
}
REGIMES = list(GRIDS)
ABL_GRIDS = {                               # E8 ablations, selected by the same rule
    'eqalign': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],          # EqAlign without the projection head
    'cosalign_ph': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],      # positives only, same head: no negatives
    'ttsimf_ph': [0.03, 0.1, 0.5, 1.0, 2.0],             # DeepGate2-faithful TT-sim
}
PAPER = {'base': None, 'aug': None, 'ttsim_ph': 0.1, 'circ_ph': 0.2, 'l1align': 0.003, 'l1tt': 0.003,
         'eqalign_ph': 0.2}
E12_HOLDOUTS = ['S1', 'S2', 'S3', 'S4', 'inj']
E12_SEEDS = [0, 1, 2]
ARCHS = {
    'E12': [('gru', 64, 2), ('gru', 256, 2)],
    'E1a': [('gru', 64, 2), ('gru', 256, 2)],
    'E1b': [('gru', 128, 2), ('gru', 512, 2), ('gru', 256, 4)],
    'E2': [('bi', 64, 2), ('bi', 128, 2), ('bi', 256, 2), ('tf', 128, 2), ('tf', 256, 2)],
    'E4': [('gru', 64, 2), ('gru', 256, 2)],
    'E8': [('gru', 64, 2), ('gru', 256, 2)],
    'E9': [('gru', 64, 2), ('gru', 256, 2)],
}
ARCHS['E15'] = [('gru', 1024, 2), ('tf', 512, 2)]
E15_REGIMES = ['base', 'aug', 'eqalign_ph']          # the three regimes the scale question needs
E15_LRS = [1e-3, 5e-4, 2.5e-4]                       # fixed in advance: E1a chose 5e-4 already at width 256
# E1c (v4.6): E1b's width-512 and 4-round encoders again, all seven regimes, with E15's lower lr grid. E1b's grid
# (2e-3 to 5e-4) left some runs of these two encoders at the constant predictor (val MAE near 0.126); the selection
# rules are unchanged, only the lr candidates follow E15.
ARCHS['E1c'] = [('gru', 512, 2), ('gru', 256, 4)]
E4_VARIANTS = [1, 2, 4]
E3_GRID = dict(lam=[0.05, 0.1, 0.2, 0.5], epochs=[5, 10])
VAL = 'val'; FULL = 'val,epfl,iscas'; FULL_H = 'val,epfl,iscas,epfl_h'


def aname(arch, d, rounds):
    return f'{arch}_d{d}_r{rounds}'


def rname(method, lam, lr, seed, V=1, tag=''):
    s = method + (f'_l{lam:g}' if lam is not None else '') + f'_lr{lr:g}'
    return s + (f'_v{V}' if V != 1 else '') + tag + f'_s{seed}'


def tjob(exp, arch, d, rounds, method, lam, lr, seed, suites, V=1, epochs=12, extra=None, tag='', probe=None, holdout=None,
         pairs=None):
    out = f'runs/{exp}/{aname(arch, d, rounds)}/{rname(method, lam, lr, seed, V, tag)}'
    args = ['--method', method, '--arch', arch, '--d', str(d), '--rounds', str(rounds), '--lr', f'{lr:g}',
            '--seed', str(seed), '--epochs', str(epochs), '--variants', str(V), '--out', out]
    if lam is not None:
        args += ['--lam', f'{lam:g}']
    if extra:
        args += ['--extra-file', extra]
    if holdout:
        args += ['--holdout', holdout] + (['--holdout-extra', extra or 'pairs_extra.pkl'] if extra else [])
    if pairs:
        args += ['--pairs-file', pairs]
    j = dict(id=out, kind='train', args=args, suites=suites, out=out)
    if probe:
        j['args'] += ['--probe-from', probe]; j['deps'] = [probe]
    return j


def dg2job(out, mode, suites, pi='shared', lam=None, epochs=None, seed=0):
    a = ['--mode', mode, '--suites', suites, '--out', out, '--seed', str(seed), '--pi', pi]
    if lam is not None:
        a += ['--lam', f'{lam:g}']
    if epochs is not None:
        a += ['--epochs', str(epochs)]
    return dict(id=out, kind='dg2', out=out, args=a)


def noisejob(exp, a, lr, seeds):
    bs = [f'runs/{exp}/{aname(*a)}/' + rname('base', None, lr, s) for s in seeds]
    return dict(id=f'runs/{exp}/{aname(*a)}/noise', kind='noise', out=f'runs/{exp}/{aname(*a)}/noise', args=bs, deps=bs)


def sel(exp):
    p = f'runs/{exp}/selection.json'
    if not os.path.exists(p):
        sys.exit(f'{p} missing: run "python -m tools.select {exp} <phase>" first')
    return json.load(open(p))


def scale_plan(exp, phase):
    archs = ARCHS[exp]; J = []
    regimes = E15_REGIMES if exp == 'E15' else REGIMES; lrs = E15_LRS if exp in ('E15', 'E1c') else LRS
    if phase == 'lr':
        for a in archs:
            for lr in lrs:
                J.append(tjob(exp, *a, 'base', None, lr, 0, VAL))
    elif phase == 'grid':
        S = sel(exp)
        for a in archs:
            lr = S[aname(*a)]['lr']
            for m in regimes:
                if m == 'base':
                    continue
                for lam in GRIDS[m]:
                    J.append(tjob(exp, *a, m, lam, lr, 0, VAL))
    elif phase == 'final':
        S = sel(exp)
        for a in archs:
            A = S[aname(*a)]
            for m in regimes:
                lam = A['lam'][m]
                Vs = E4_VARIANTS if (exp == 'E4' and m != 'base') else [1]
                for V in Vs:
                    for s in SEEDS:
                        suites = FULL_H if m == 'eqalign_ph' else FULL
                        J.append(tjob(exp, *a, m, lam, A['lr'], s, suites, V=V))
            J.append(noisejob(exp, a, A['lr'], SEEDS))
    else:
        sys.exit(f'unknown phase {phase} for {exp}')
    return J


def plan(exp, phase):
    J = []
    if exp == 'smoke':
        for m in REGIMES:
            J.append(tjob('smoke', 'gru', 32, 2, m, PAPER[m], 2e-3, 0, FULL_H if m == 'eqalign_ph' else FULL, epochs=1))
        for m, lam in (('eqalign', 0.2), ('cosalign_ph', 0.2), ('ttsimf_ph', 0.1)):
            J.append(tjob('smoke', 'gru', 32, 2, m, lam, 2e-3, 0, FULL, epochs=1))
        b0 = 'runs/smoke/gru_d32_r2/' + rname('base', None, 2e-3, 0)
        J.append(tjob('smoke', 'gru', 32, 2, 'eqalign_ph', 0.2, 2e-3, 0, FULL, epochs=1, tag='_probe', probe=b0))
        J.append(tjob('smoke', 'bi', 32, 2, 'eqalign_ph', 0.2, 2e-3, 0, FULL, epochs=1))
        J.append(tjob('smoke', 'tf', 32, 2, 'eqalign_ph', 0.2, 2e-3, 0, FULL, epochs=1))
        J.append(tjob('smoke', 'gru', 32, 2, 'eqalign_ph', 0.2, 2e-3, 0, FULL, V=2, epochs=1, extra='pairs_extra_smoke.pkl'))
        J.append(tjob('smoke', 'gru', 32, 2, 'aug', None, 2e-3, 0, FULL, V=2, epochs=1, extra='pairs_h2_smoke.pkl', tag='_h2'))
        J.append(tjob('smoke', 'gru', 32, 2, 'eqalign_ph', 0.2, 2e-3, 0, FULL, epochs=1, extra='pairs_extra_smoke.pkl', tag='_hoS1', holdout='S1'))
        J.append(dg2job('runs/smoke/dg2_released', 'released', FULL))
        J.append(dg2job('runs/smoke/dg2_ctrl', 'control', FULL, pi='random', epochs=1))
        J.append(dg2job('runs/smoke/dg2_eq', 'eqalign', FULL, pi='shared', lam=0.2, epochs=1))
        J.append(dict(id='runs/smoke/noise', kind='noise', out='runs/smoke/noise', args=[b0, b0], deps=[b0]))
        e0 = 'runs/smoke/gru_d32_r2/' + rname('eqalign_ph', 0.2, 2e-3, 0)
        J.append(dict(id='runs/smoke/sweep', kind='sweep', out=e0, args=['--run', e0], deps=[e0]))
        r = 'runs/smoke/dg2_scratch'
        J.append(dict(id=r + '/retrain', kind='retrain', out=r, args=['--seed', '0', '--out', r, '--epochs', '1', '--limit', '40']))
        j = dg2job(r + '/audit', 'released', VAL); j['args'] += ['--ckpt', r + '/model.pth']; j['deps_file'] = [r + '/model.pth']
        J.append(j)
        return J
    if exp == 'E0':
        for m in REGIMES:
            for s in (0, 1, 2):
                J.append(tjob('E0', 'gru', 64, 2, m, PAPER[m], 2e-3, s, FULL))
        J.append(noisejob('E0', ('gru', 64, 2), 2e-3, (0, 1, 2)))
        return J
    if exp in ('E1a', 'E1b', 'E1c', 'E2', 'E4', 'E15'):
        return scale_plan(exp, phase)
    if exp == 'E8':
        S1 = sel('E1a')
        if phase == 'grid':
            for a in ARCHS['E8']:
                lr = S1[aname(*a)]['lr']
                for m, g in ABL_GRIDS.items():
                    for lam in g:
                        J.append(tjob('E8', *a, m, lam, lr, 0, VAL))
        elif phase == 'final':
            S = sel('E8')
            for a in ARCHS['E8']:
                A1 = S1[aname(*a)]; lr = A1['lr']
                for m in ABL_GRIDS:
                    for s in SEEDS:
                        J.append(tjob('E8', *a, m, S[aname(*a)]['lam'][m], lr, s, FULL_H if m == 'eqalign' else FULL))
                lam_eq = A1['lam']['eqalign_ph']           # probes: EqAlign's own selected weight, frozen encoders
                for src in ('base', 'aug'):
                    for s in SEEDS:
                        probe = f'runs/E1a/{aname(*a)}/' + rname(src, None, lr, s)
                        J.append(tjob('E8', *a, 'eqalign_ph', lam_eq, lr, s, FULL, tag=f'_probe{src}', probe=probe))
        else:
            sys.exit(f'unknown phase {phase} for E8')
        return J
    if exp == 'E9':                         # H2 positive control: same weights as E1a, H2 variants added
        S1 = sel('E1a')
        for a in ARCHS['E9']:
            A1 = S1[aname(*a)]
            for m in ('aug', 'eqalign_ph'):
                for s in SEEDS:
                    J.append(tjob('E9', *a, m, A1['lam'][m], A1['lr'], s, FULL, V=2, extra='pairs_h2.pkl', tag='_h2'))
        return J
    if exp == 'E12':                        # leave-one-family-out; weights and lr from E1a, nothing re-tuned
        S1 = sel('E1a')
        for a in ARCHS['E12']:
            A1 = S1[aname(*a)]
            for F in E12_HOLDOUTS:
                for m in ('aug', 'eqalign_ph'):
                    for s in E12_SEEDS:
                        J.append(tjob('E12', *a, m, A1['lam'][m], A1['lr'], s, FULL, tag=f'_ho{F}', holdout=F))
        return J
    if exp == 'E3':
        if phase == 'grid':                 # primary protocol: shared PI vectors during fine-tuning
            for ep in E3_GRID['epochs']:
                J.append(dg2job(f'runs/E3/grid/dg2_control_e{ep}_s0', 'control', VAL, epochs=ep))
                for lam in E3_GRID['lam']:
                    J.append(dg2job(f'runs/E3/grid/dg2_eqalign_l{lam:g}_e{ep}_s0', 'eqalign', VAL, lam=lam, epochs=ep))
        elif phase == 'final':
            S = sel('E3')
            J.append(dg2job('runs/E3/dg2_released', 'released', FULL))
            for pi, settings in (('shared', {(S['lam'], S['epochs']), (0.2, 5)}), ('random', {(0.2, 5)})):
                for lam, ep in sorted(settings):
                    for s in SEEDS:
                        J.append(dg2job(f'runs/E3/final/pi_{pi}/dg2_eqalign_l{lam:g}_e{ep}_s{s}', 'eqalign', FULL, pi, lam, ep, s))
                for ep in sorted({ep for _, ep in settings}):
                    for s in SEEDS:
                        J.append(dg2job(f'runs/E3/final/pi_{pi}/dg2_control_e{ep}_s{s}', 'control', FULL, pi, None, ep, s))
        else:
            sys.exit(f'unknown phase {phase} for E3')
        return J
    if exp == 'E10':
        for s in (0, 1, 2):
            r = f'runs/E10/dg2_scratch_s{s}'
            J.append(dict(id=r + '/retrain', kind='retrain', out=r, args=['--seed', str(s), '--out', r]))
            j = dg2job(r + '/audit', 'released', FULL); j['args'] += ['--ckpt', r + '/model.pth']; j['deps_file'] = [r + '/model.pth']
            J.append(j)
        return J
    if exp == 'E14':                        # supplement; weights, lr and models from E1a, nothing re-tuned
        S1 = sel('E1a')
        for a in ARCHS['E1a']:
            A1 = S1[aname(*a)]
            for bits in (64, 4096):
                for s in SEEDS:
                    J.append(tjob('E14', *a, 'eqalign_ph', A1['lam']['eqalign_ph'], A1['lr'], s, FULL,
                                  tag=f'_sig{bits}', pairs=f'pairs_sig{bits}.pkl'))
            for m in REGIMES:                   # (b) hyp: evaluation of the existing E1a final models only
                for s in SEEDS:
                    j = tjob('E1a', *a, m, A1['lam'][m], A1['lr'], s, 'hyp')
                    j['id'] = j['id'] + '#hyp'; j['deps'] = [j['out']]
                    J.append(j)
            for m in REGIMES:                   # (d) Table II: test-set evaluation of the weight-grid runs (seed 0)
                for lam in GRIDS[m]:            # that were not selected; selection was fixed on validation already
                    if m == 'base' or lam == A1['lam'][m]:
                        continue
                    j = tjob('E1a', *a, m, lam, A1['lr'], 0, 'epfl')
                    j['id'] = j['id'] + '#test'; j['deps'] = [j['out']]
                    J.append(j)
            for m in ('base', 'aug', 'ttsim_ph', 'circ_ph', 'eqalign_ph'):   # (c) exports
                for s in SEEDS:
                    run = f'runs/E1a/{aname(*a)}/' + rname(m, A1['lam'][m], A1['lr'], s)
                    J.append(dict(id=run + '#export', kind='export', out=run, deps=[run],
                                  args=['--run', run] + (['--fig1'] if m in ('base', 'eqalign_ph') else [])))
        return J
    if exp in ('E17', 'E17smoke'):          # ABC rewrite families: evaluation of finished models only, nothing trained
        J = []
        suites = 'abc,abc_iscas'
        exps = ['smoke'] if exp == 'E17smoke' else os.environ.get('E17_EXPS', 'E1a,E2,E15,E3,E10').split(',')
        for e in exps:
            for tj in sorted(glob.glob(f'runs/{e}/**/train.json', recursive=True)):
                run = os.path.dirname(tj)
                if not os.path.exists(os.path.join(run, 'raw_epfl.pkl')) and e != 'smoke':
                    continue                    # lr / weight-grid runs (validation only) are not test-evaluated
                c = json.load(open(tj))['config']; m = c.get('method', '')
                if m.startswith('dg2'):
                    a = ['--mode', m[4:], '--suites', suites, '--out', run]
                    if c.get('ckpt'):
                        a += ['--ckpt', c['ckpt']]
                    J.append(dict(id=run + '#abc', kind='dg2eval', out=run, args=a, deps_file=[tj]))
                elif os.path.exists(os.path.join(run, 'model.pt')):
                    J.append(dict(id=run + '#abc', kind='train', out=run, args=[], suites=suites, deps_file=[tj]))
        return J
    if exp == 'E5':
        for e in os.environ.get('E5_EXPS', 'E1a,E1b').split(','):   # E5_EXPS=E1a: only the main-text models
            if not os.path.exists(f'runs/{e}/selection.json'):
                continue
            S = sel(e)
            for a in ARCHS[e]:
                A = S[aname(*a)]
                for m in ('base', 'aug', 'ttsim_ph', 'circ_ph', 'eqalign_ph'):
                    for s in (0, 1, 2):
                        run = f'runs/{e}/{aname(*a)}/{rname(m, A["lam"][m], A["lr"], s)}'
                        J.append(dict(id=run + '/sweep', kind='sweep', out=run, args=['--run', run], deps=[run]))
        return J
    sys.exit(f'unknown experiment {exp}')


if __name__ == '__main__':
    exp = sys.argv[1]; phase = sys.argv[2] if len(sys.argv) > 2 else 'all'
    J = plan(exp, phase)
    ids = [j['id'] for j in J]
    assert len(ids) == len(set(ids)), 'duplicate job ids'
    os.makedirs('jobs', exist_ok=True)
    path = f'jobs/{exp}_{phase}.jsonl'
    with open(path, 'w') as f:
        for j in J:
            f.write(json.dumps(j) + '\n')
    print(f'{path}: {len(J)} jobs')
