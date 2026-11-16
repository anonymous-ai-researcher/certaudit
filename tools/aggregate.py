"""Aggregate finished runs into tables, statistical tests and the pre-declared claims check (README section 7).

  python -m tools.aggregate E0 | E1 | E2 | E3 | E4 | E8 | E9 | E12 | E13 | E5 | all       (E1 = E1a + E1b)

Writes results/<exp>/:
  summary.csv       one row per configuration, evaluation protocol and suite (EPFL / ISCAS): mean, s.d.
                    (ddof=1), n seeds, n diverged (excluded) and a 95% t-interval for every metric of the paper's
                    Table I, per-family splits (S1..S4, I1/I3/I5) and held-out H1/H2, all-pair metrics
  comparisons.csv   design-level differences (EPFL: 7 test designs; ISCAS: 17 designs), cones and seeds pooled:
                    mean, 95% cluster-bootstrap CI, t-interval over designs, designs better, sign test, exact
                    sign-flip permutation test, Wilcoxon; seed-level Welch t-test; Holm-adjusted p
  trends.csv        width trends (slope per doubling, bootstrap CI over seeds; Spearman of width means)
  claims.md         every pre-declared claim with PASS / FAIL / INCOMPLETE and the numbers behind it
  report.md         readable tables
Only runs with raw_epfl.pkl enter (validation-only grid runs never do). Diverged runs are excluded and counted."""
import csv, glob, json, os, pickle, sys, numpy as np
from core.stats import METRICS, seed_values, mean_sd_ci, design_test, welch, holm, trend, PER_CIRCUIT

NAMES = {'base': 'Baseline', 'aug': 'Augmentation', 'ttsim_ph': 'TT-sim', 'circ_ph': 'Circuit contrast',
         'l1align': 'L1 alignment', 'l1tt': 'L1 + TT-sim', 'eqalign_ph': 'EqAlign', 'eqalign': 'EqAlign (no head)',
         'cosalign_ph': 'Cosine alignment (positives only)', 'ttsimf_ph': 'TT-sim (DeepGate2 filters)',
         'dg2_released': 'DeepGate2 released', 'dg2_control': 'DG2 fine-tuned (control)', 'dg2_eqalign': 'DG2 fine-tuned + EqAlign'}
OTHERS = ['base', 'aug', 'ttsim_ph', 'circ_ph', 'l1align', 'l1tt']
KEY = (['MAE', 'dMAE_syn', 'dMAE_inj', 'dMAE_H1', 'dMAE_H2', 'rho_syn', 'rho_inj', 'rho_H1', 'rho_H2', 'rho_all',
        'rho_allpairs_syn', 'rho_allpairs_inj', 'gap_syn', 'gap_inj', 'incons_rs', 'incons_rs_syn', 'incons_rs_inj',
        'top1_seen', 'top1_syn', 'top1_inj', 'top1_ho', 'top1_H1', 'top1_H2', 'top1_all', 'auroc_lvl', 'worst_ratio',
        'chance'] + [m for m in METRICS if m[-2:] in ('S1', 'S2', 'S3', 'S4', 'I1', 'I3', 'I5')])
CMP = ['top1_seen', 'top1_ho', 'top1_all', 'top1_H2', 'incons_rs', 'incons_rs_inj', 'MAE', 'dMAE_H2', 'auroc_lvl']
EXPDIRS = {'E1': ['E1a', 'E1b']}
_RAW = {}


def raw(p):
    if p not in _RAW:
        _RAW[p] = pickle.load(open(p, 'rb'))
    return _RAW[p]


# ------------------------------------------------------------------------------------------------ collection
def collect(exp):
    """config key -> dict(cfg, arch, files={suite: [paths]}, n_diverged). Key fields:
    (arch, method, lam, lr, V, tag, eval) where tag marks probes / H2 training and eval the evaluation protocol."""
    G = {}
    for e in EXPDIRS.get(exp, [exp]):
        for tj in sorted(glob.glob(f'runs/{e}/**/train.json', recursive=True)):
            run = os.path.dirname(tj); c = json.load(open(tj))['config']
            if 'grid' in run.split(os.sep):
                continue
            if c.get('method', '').startswith('dg2'):
                arch = 'deepgate2'; lam = c.get('lam') if c['method'] == 'dg2_eqalign' else None
                ep = c.get('epochs') if c['method'] != 'dg2_released' else None
                tag = f"train_pi={c.get('pi')}" if c['method'] != 'dg2_released' else ('retrained from scratch' if c.get('ckpt') else '')
                evals = [('shared', run), ('shared_pi1', f'{run}/eval_shared_pi1'), ('shared_pi2', f'{run}/eval_shared_pi2'),
                         ('random', f'{run}/eval_random'), ('hf', f'{run}/raw_hf')]
                base_key = (arch, c['method'], lam, ep, 1, tag)
            else:
                arch = f"{c['arch']}_d{c['d']}_r{c['rounds']}"
                lam = c.get('lam') if c['method'] not in ('base', 'aug') else None
                tag = ('probe:' + ('base' if '/base_' in c['probe_from'] else 'aug')) if c.get('probe_from') else \
                      (f"holdout:{c['holdout']}" if c.get('holdout') else
                       ('h2-trained' if 'h2' in str(c.get('extra_file', '')) and c.get('variants', 1) > 1 else ''))
                V = c.get('variants', 1) if not tag.startswith('h2') else 1
                evals = [('z', run)] + ([('h', run)] if os.path.exists(f'{run}/raw_epfl_h.pkl') else [])
                base_key = (arch, c['method'], lam, c['lr'], V, tag)
            for ev, d in evals:
                fn = 'raw_epfl_h.pkl' if ev == 'h' else 'raw_epfl.pkl'
                if not os.path.exists(os.path.join(d, fn)):
                    continue
                g = G.setdefault(base_key + (ev,), dict(cfg=c, arch=arch, files={'epfl': [], 'iscas': [], 'large': []}, n_diverged=0, runs=[]))
                if c.get('diverged'):
                    g['n_diverged'] += 1; continue
                g['files']['epfl'].append(os.path.join(d, fn)); g['runs'].append(run)
                if ev in ('z', 'shared') and os.path.exists(os.path.join(d, 'raw_iscas.pkl')):
                    g['files']['iscas'].append(os.path.join(d, 'raw_iscas.pkl'))
                if ev == 'z' and os.path.exists(os.path.join(d, 'raw_large.pkl')):
                    g['files']['large'].append(os.path.join(d, 'raw_large.pkl'))
    return G


def label(k):
    arch, m, lam, lr_or_ep, V, tag, ev = k
    s = NAMES.get(m, m) + (f' lambda={lam:g}' if lam is not None else '')
    if arch == 'deepgate2' and lr_or_ep is not None:
        s += f', {lr_or_ep} ep'
    s += (f', V={V}' if V != 1 else '') + (f', {tag}' if tag else '') + (f' [eval {ev}]' if ev not in ('z', 'shared') else '')
    return s


def R_of(g, suite='epfl'):
    return [raw(f) for f in g['files'][suite]]


# ------------------------------------------------------------------------------------------------ tables
def summarize(G, exp, suite):
    rows = []
    for k, g in sorted(G.items(), key=lambda kv: str(kv[0])):
        R = R_of(g, suite)
        if not R:
            continue
        row = dict(exp=exp, suite=suite, arch=k[0], method=k[1], lam=k[2], variants=k[4], tag=k[5], eval=k[6],
                   config=label(k), n_seeds=len(R), n_diverged=g['n_diverged'])
        for m in KEY:
            s = mean_sd_ci(seed_values(R, m))
            row[f'{m}_mean'], row[f'{m}_sd'], row[f'{m}_ci_lo'], row[f'{m}_ci_hi'] = s['mean'], s['sd'], s['ci_lo'], s['ci_hi']
        rows.append(row)
    return rows


def cmp_rows(gB, gA, nameB, nameA, ctx, suite='epfl', mets=None):
    Ra, Rb = R_of(gA, suite), R_of(gB, suite)
    if not Ra or not Rb:
        return []
    out = []
    for met in (mets or CMP):
        d = design_test(Ra, Rb, met)
        pw = welch(seed_values(Ra, met), seed_values(Rb, met)) if met in METRICS else float('nan')
        out.append(dict(ctx, suite=suite, B=nameB, A=nameA, metric=met, diff_B_minus_A=d['diff'], ci_lo=d['ci_lo'],
                        ci_hi=d['ci_hi'], t_lo=d['t_lo'], t_hi=d['t_hi'], designs_better=f"{d['n_better']}/{d['n_designs']}",
                        p_sign=d['p_sign'], p_perm=d['p_perm'], p_wilcoxon=d['p_wilcoxon'], p_welch_seeds=pw,
                        n_seeds_A=len(Ra), n_seeds_B=len(Rb)))
    return out


def add_holm(rows, Bname='EqAlign'):
    keyf = lambda r: (r.get('arch'), r.get('variants'), r.get('suite'), r['metric'])
    groups = {}
    for i, r in enumerate(rows):
        if r['B'] == Bname:
            groups.setdefault(keyf(r), []).append(i)
    for idx in groups.values():
        for col in ('p_sign', 'p_perm', 'p_welch_seeds'):
            ps = [rows[i][col] if rows[i][col] == rows[i][col] else 1.0 for i in idx]
            for i, q in zip(idx, holm(ps)):
                rows[i][col + '_holm'] = q
    return rows


def by(G, **f):
    names = ['arch', 'method', 'lam', 'lr', 'V', 'tag', 'eval']
    return {k: g for k, g in G.items() if all(k[names.index(n)] == v for n, v in f.items())}


def one(G, **f):
    m = by(G, **f)
    return next(iter(m.values())) if len(m) == 1 else None


def comparisons(G, exp, ref=None):
    C = []
    if exp == 'E10':
        return []
    if exp == 'E3':
        for k, g in by(G, method='dg2_eqalign').items():
            ev, ep, tag = k[6], k[3], k[5]
            ctl = one(G, method='dg2_control', lr=ep, tag=tag, eval=ev)
            rl = one(G, method='dg2_released', eval=ev if ev != 'hf' else 'shared')
            ctx = dict(arch='deepgate2', variants=1, setting=label(k))
            if ctl:
                C += cmp_rows(g, ctl, label(k), label(by(G, method='dg2_control', lr=ep, tag=tag, eval=ev).popitem()[0]), ctx)
            if rl:
                C += cmp_rows(g, rl, label(k), 'DeepGate2 released', ctx)
        return C
    archs = sorted(set(k[0] for k in G))
    for arch in archs:
        Vs = sorted(set(k[4] for k in by(G, arch=arch, tag='', eval='z')))
        for V in Vs:
            M = {k[1]: g for k, g in by(G, arch=arch, V=V, tag='', eval='z').items()}
            if V != 1 and 'base' not in M:
                b = one(G, arch=arch, method='base', V=1, tag='', eval='z')
                if b:
                    M['base'] = b
            ctx = dict(arch=arch, variants=V)
            for suite in ('epfl', 'iscas', 'large'):
                if 'eqalign_ph' in M:
                    for o in OTHERS:
                        if o in M:
                            C += cmp_rows(M['eqalign_ph'], M[o], 'EqAlign', NAMES[o], ctx, suite)
                if 'aug' in M and 'base' in M:
                    C += cmp_rows(M['aug'], M['base'], 'Augmentation', 'Baseline', ctx, suite)
                if 'l1align' in M and 'aug' in M:
                    C += cmp_rows(M['l1align'], M['aug'], 'L1 alignment', 'Augmentation', ctx, suite)
    if exp == 'E4':                       # data scale: V=2,4 vs V=1 per regime; width at V=1 and V=4
        for arch in archs:
            for m in set(k[1] for k in by(G, arch=arch)):
                g1 = one(G, arch=arch, method=m, V=1, tag='', eval='z')
                for V in (2, 4):
                    gv = one(G, arch=arch, method=m, V=V, tag='', eval='z')
                    if g1 and gv:
                        C += cmp_rows(gv, g1, f'{NAMES[m]} V={V}', f'{NAMES[m]} V=1', dict(arch=arch, variants=V, contrast='data'))
        for m in ('aug', 'eqalign_ph', 'l1align'):
            for V in (1, 4):
                a = one(G, arch='gru_d64_r2', method=m, V=V, tag='', eval='z'); b = one(G, arch='gru_d256_r2', method=m, V=V, tag='', eval='z')
                if a and b:
                    C += cmp_rows(b, a, f'{NAMES[m]} d256 V={V}', f'{NAMES[m]} d64 V={V}', dict(arch='d256 vs d64', variants=V, contrast='width'))
    if exp == 'E12' and ref:              # leave-one-family-out, metrics on the held-out test family only
        for arch in sorted(set(k[0] for k in G)):
            R = {k[1]: g for k, g in by(ref, arch=arch, tag='', eval='z').items()}
            for F in E12_FAMS:
                mets = E12_METS(F); ctx = dict(arch=arch, variants=1, holdout=F)
                eq = one(G, arch=arch, method='eqalign_ph', tag=f'holdout:{F}', eval='z')
                au = one(G, arch=arch, method='aug', tag=f'holdout:{F}', eval='z')
                if eq and au:
                    C += cmp_rows(eq, au, f'EqAlign without {F}', f'Augmentation without {F}', ctx, mets=mets)
                for m, g in (('eqalign_ph', eq), ('aug', au)):
                    if g and R.get(m):
                        C += cmp_rows(g, R[m], f'{NAMES[m]} without {F}', NAMES[m], ctx, mets=mets)
                if eq and R.get('base'):
                    C += cmp_rows(eq, R['base'], f'EqAlign without {F}', 'Baseline', ctx, mets=mets)
        return C                          # Holm is applied per held-out family inside claims_e12
    if exp in ('E8', 'E9') and ref:
        for arch in sorted(set(k[0] for k in G)):
            R = {k[1]: g for k, g in by(ref, arch=arch, tag='', eval='z').items()}
            ctx = dict(arch=arch, variants=1)
            if exp == 'E8':
                eq = R.get('eqalign_ph')
                for k, g in by(G, arch=arch).items():
                    if eq and k[6] == 'z':
                        C += cmp_rows(eq, g, 'EqAlign', label(k), ctx)
                cos = one(G, arch=arch, method='cosalign_ph', eval='z')
                if cos and R.get('aug'):
                    C += cmp_rows(cos, R['aug'], 'Cosine alignment (positives only)', 'Augmentation', ctx)
                pb = one(G, arch=arch, tag='probe:base', eval='z')
                if pb and R.get('base'):
                    C += cmp_rows(pb, R['base'], 'probe on frozen Baseline', 'Baseline', ctx)
                eh = one(ref, arch=arch, method='eqalign_ph', tag='', eval='h')
                if eq and eh:
                    C += cmp_rows(eh, eq, 'EqAlign, retrieval on h', 'EqAlign, retrieval on z', ctx)
            else:
                for m in ('aug', 'eqalign_ph'):
                    g = one(G, arch=arch, method=m, tag='h2-trained', eval='z')
                    if g and R.get(m):
                        C += cmp_rows(g, R[m], f'{NAMES[m]} trained with H2', NAMES[m], ctx)
    return add_holm(C)


def trends(G, exp):
    out, fam = [], {}
    for k, g in by(G, V=1, tag='', eval='z').items():
        c = g['cfg']
        if c.get('rounds') == 2:
            fam.setdefault((c['arch'], k[1]), []).append((c['d'], R_of(g)))
    for (a, m), L in sorted(fam.items()):
        if len(L) < 2:
            continue
        for met in ['top1_seen', 'top1_all', 'top1_ho', 'top1_H2', 'rho_syn', 'rho_inj', 'rho_allpairs_syn', 'incons_rs',
                    'MAE', 'dMAE_H2', 'rho_H2']:
            xs, ys = [], []
            for d, R in L:
                v = seed_values(R, met); xs += [d] * len(v); ys += v.tolist()
            out.append(dict(exp=exp, family=a, regime=NAMES[m], metric=met, widths='/'.join(str(d) for d, _ in sorted(L)), **trend(xs, ys)))
    return out


# ------------------------------------------------------------------------------------------------ claims
def fmt(m, sd, k):
    if m != m:
        return 'n/a'
    if k.startswith(('top1', 'gap', 'chance')):
        return f'{m:.1f} ± {sd:.1f}'
    if k.startswith(('rho', 'worst', 'auroc')):
        return f'{m:.2f} ± {sd:.2f}'
    return f'{m:.4f} ± {sd:.4f}'


def find(C, **f):
    return next((c for c in C if all(c.get(k) == v for k, v in f.items())), None)


def full(c):
    a, b = c['designs_better'].split('/'); return a == b


def claims(S, C, exp, noise, suite='epfl'):
    L = [f'# Claims check, {exp}\n', 'Rules: README section 7 (fixed before any run). PASS needs every input present; '
         'missing inputs give INCOMPLETE. Numbers: mean ± s.d. over seeds; design-level differences [95% cluster-bootstrap CI].\n']
    for arch in sorted(set(r['arch'] for r in S if r['suite'] == suite and r['tag'] == '' and r['eval'] == 'z')):
        allpairs = arch.startswith(('bi_', 'tf_'))
        rho_k, top_k = ('rho_allpairs_syn', 'top1_all') if allpairs else ('rho_syn', 'top1_seen')
        for V in sorted(set(r['variants'] for r in S if r['arch'] == arch)):
            R = {r['method']: r for r in S if r['suite'] == suite and r['arch'] == arch and r['tag'] == '' and r['eval'] == 'z'
                 and (r['variants'] == V or (r['method'] == 'base' and r['variants'] == 1))}
            L.append(f'\n## {arch}' + (f', {V} certified variants per cone' if V != 1 else '') +
                     (' (all-pair metrics, as the paper did for the bidirectional encoder)' if allpairs else '') + '\n')
            div = sum(r['n_diverged'] for r in R.values())
            if div:
                L.append(f'- NOTE: {div} diverged run(s) excluded (see summary.csv n_diverged)')
            b = R.get('base')
            if b is None:
                L.append('- C1 to C5: **INCOMPLETE** (no baseline)'); continue
            nf = noise.get(arch)
            if nf is None and suite == 'large':   # E13: no seed-noise floor on this suite; the rho part of the rule only
                ok = b[rho_k + '_mean'] > 0.5
                L.append(f"- **C1 equivalence blindness (large circuits)**: {rho_k} = {fmt(b[rho_k + '_mean'], b[rho_k + '_sd'], 'rho')}, "
                         f"{top_k} = {fmt(b[top_k + '_mean'], b[top_k + '_sd'], 'top1')}%. Rule: rho > 0.5 -> **{'PASS' if ok else 'FAIL'}**")
            elif nf is None:
                L.append(f"- **C1**: rho = {fmt(b[rho_k + '_mean'], b[rho_k + '_sd'], 'rho')}; noise floor missing -> **INCOMPLETE**")
            else:
                ratio = b['incons_rs_syn_mean'] / nf; ok = b[rho_k + '_mean'] > 0.5 and ratio > 2
                L.append(f"- **C1 equivalence blindness**: {rho_k} = {fmt(b[rho_k + '_mean'], b[rho_k + '_sd'], 'rho')}, synthesis "
                         f"inconsistency / seed noise floor = {ratio:.1f}x, {top_k} = {fmt(b[top_k + '_mean'], b[top_k + '_sd'], 'top1')}% "
                         f"(chance {b['chance_mean']:.1f}%). Rule: rho > 0.5 and ratio > 2 -> **{'PASS' if ok else 'FAIL'}**")
            c = find(C, arch=arch, variants=V, suite=suite, B='Augmentation', A='Baseline', metric='incons_rs')
            L.append('- **C2 augmentation steadies predictions**: ' + ('**INCOMPLETE**' if not c else
                     f"Aug - Base inconsistency {c['diff_B_minus_A']:.4f} [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}], t [{c['t_lo']:.4f}, {c['t_hi']:.4f}], "
                     f"better on {c['designs_better']}, sign-flip p {c['p_perm']:.3f}. Rule: bootstrap CI and t-interval below 0 -> "
                     f"**{'PASS' if c['ci_hi'] < 0 and c['t_hi'] < 0 else 'FAIL'}**"))
            if 'eqalign_ph' in R:
                parts, fails, miss = [], [], [NAMES[o] for o in OTHERS if o not in R]
                for o in OTHERS:
                    c = find(C, arch=arch, variants=V, suite=suite, B='EqAlign', A=NAMES[o], metric=top_k)
                    if not c:
                        continue
                    ok = c['ci_lo'] > 0 and full(c)
                    parts.append(f"vs {NAMES[o]}: {c['diff_B_minus_A']:+.1f} pts [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}], {c['designs_better']} designs, "
                                 f"Welch p (Holm) {c.get('p_welch_seeds_holm', float('nan')):.3g}")
                    fails += [] if ok else [NAMES[o]]
                verdict = 'INCOMPLETE (missing: ' + ', '.join(miss) + ')' if miss else ('PASS' if not fails else 'FAIL (' + ', '.join(fails) + ')')
                L.append(f'- **C3 only EqAlign repairs the representation** ({top_k}):\n  - ' + '\n  - '.join(parts) +
                         f'\n  Rule: CI above 0 and better on every design against every other regime -> **{verdict}**')
            rows = [(m, R[m]['dMAE_H2_mean'], R[m]['rho_H2_mean']) for m in R]
            ok = all(dm >= 0.05 and rh >= 2 for _, dm, rh in rows)
            L.append('- **C4 no tested repair survives H2**: ' + '; '.join(f"{NAMES[m]} dMAE {dm:.3f}, rho {rh:.1f}" for m, dm, rh in rows) +
                     f". Rule: every regime dMAE_H2 >= 0.05 and rho_H2 >= 2 -> **{'PASS (all fail on H2)' if ok else 'FAIL (a regime survives H2: revise the paper)'}**")
            c = find(C, arch=arch, variants=V, suite=suite, B='EqAlign', A='Baseline', metric='MAE')
            if c:
                L.append(f"- **C5 accuracy cost**: MAE EqAlign - Base {c['diff_B_minus_A']:.4f} [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}] "
                         f"({100 * (R['eqalign_ph']['MAE_mean'] / b['MAE_mean'] - 1):+.0f}%). Reported, no rule; note the 25% "
                         'validation-MAE tolerance of the selection rule bounds this cost by design.')
    return L


def claims_e3(S, C, exp='E3'):
    L = [f'# Claims check, {exp} (DeepGate2)\n']
    rel = {r['eval']: r for r in S if r['suite'] == 'epfl' and r['method'] == 'dg2_released'}
    sc = [json.load(open(f)) | {'run': f} for f in glob.glob(f'runs/{exp}/**/selfcheck.json', recursive=True)]
    for f in sc:
        if f['run'].startswith(('runs/E3/dg2_released', 'runs/E10/')):
            L.append(f"- Self-check, {f['run'].split('/selfcheck')[0]}: same circuit vs itself, Top-1 shared {100 * f['shared']['self_top1']:.1f}%, "
                     f"random {100 * f['random']['self_top1']:.1f}% (|dp| {f['random']['self_abs_dp']:.4f}). The random protocol is "
                     'not a valid audit protocol if its self Top-1 is far below 100%.')
    for ev in ('shared', 'random'):
        r = rel.get(ev)
        if r:
            L.append(f"- **C1-DG2 ({ev} PI protocol)**: released rho_syn {fmt(r['rho_syn_mean'], 0, 'rho')}, Top-1 seen {r['top1_seen_mean']:.1f}%, "
                     f"H2 dMAE {r['dMAE_H2_mean']:.3f}, rho_H2 {r['rho_H2_mean']:.1f} -> rho_syn > 0.5: **{'PASS' if r['rho_syn_mean'] > 0.5 else 'FAIL'}**")
    for c in C:
        if c['metric'] == 'top1_seen' and c['A'].startswith('DG2 fine-tuned (control)'):
            ok = c['ci_lo'] > 0 and full(c)
            L.append(f"- **C6** {c['B']} vs {c['A']}: {c['diff_B_minus_A']:+.1f} pts [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}], {c['designs_better']} designs, "
                     f"Welch p {c['p_welch_seeds']:.3g} -> **{'PASS' if ok else 'FAIL'}**")
    for c in C:
        if c['metric'] == 'MAE' and c['A'].startswith('DG2 fine-tuned (control)'):
            L.append(f"  - MAE cost {c['B']}: {c['diff_B_minus_A']:+.4f} [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]")
    return L


def claims_e8_e9(C, exp):
    L = [f'# Claims check, {exp}\n']
    for arch in sorted(set(c['arch'] for c in C)):
        L.append(f'\n## {arch}\n')
        if exp == 'E8':
            top = [x for x in C if x['arch'] == arch and x['metric'] == 'top1_seen' and x['suite'] == 'epfl']
            c = next((x for x in top if x['B'] == 'EqAlign' and x['A'].startswith('Cosine')), None)
            L.append('- **C7 negatives are needed** (EqAlign minus positives-only cosine alignment through the same head, Top-1): ' +
                     ('**INCOMPLETE**' if not c else f"{c['diff_B_minus_A']:+.1f} [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}], {c['designs_better']} -> "
                      f"**{'PASS' if c['ci_lo'] > 0 and full(c) else 'FAIL'}**"))
            L.append('- Other mechanism contrasts (Top-1, reported without a rule):')
            for x in top:
                if x is not c:
                    L.append(f"  - {x['B']} minus {x['A']}: {x['diff_B_minus_A']:+.1f} [{x['ci_lo']:.1f}, {x['ci_hi']:.1f}], {x['designs_better']}")
        else:
            for m in ('EqAlign', 'Augmentation'):
                for met in ('top1_H2', 'dMAE_H2', 'top1_seen', 'MAE'):
                    x = find(C, arch=arch, B=f'{m} trained with H2', A=m, metric=met)
                    if x:
                        L.append(f"- {m} trained with H2 minus {m}, {met}: {x['diff_B_minus_A']:+.4f} [{x['ci_lo']:.4f}, {x['ci_hi']:.4f}], {x['designs_better']}")
            x = find(C, arch=arch, B='EqAlign trained with H2', A='EqAlign', metric='dMAE_H2')
            L.append('- **C8 H2 is learnable when seen in training** (H2 matched-gate error falls): ' + ('**INCOMPLETE**' if not x else
                     f"**{'PASS' if x['ci_hi'] < 0 else 'FAIL'}**") + '. Descriptive: it tells whether the H2 failure is a generalization '
                     'failure (PASS) or H2 is hard even when trained on (FAIL).')
    return L


E12_FAMS = ['S1', 'S2', 'S3', 'S4', 'inj']
E12_METS = lambda F: [f'top1_{F}', f'incons_rs_{F}', f'dMAE_{F}']


def claims_e12(C):
    """C9 (pre-declared): does EqAlign's retrieval gain over Augmentation survive when the test family was never
    seen in training? Per held-out family F, PASS if the design-level CI of EqAlign-without-F minus
    Augmentation-without-F on Top-1 (restructured pairs of family F) excludes 0 and every design agrees.
    C9b (descriptive): the generalization gap, EqAlign-without-F minus EqAlign trained with F, on the same metric."""
    L = ['# Claims check, E12 (leave-one-family-out)\n']
    for arch in sorted(set(c['arch'] for c in C)):
        L.append(f'\n## {arch}\n')
        npass = 0
        for F in E12_FAMS:
            m = f'top1_{F}'
            x = find(C, arch=arch, B=f'EqAlign without {F}', A=f'Augmentation without {F}', metric=m)
            g = find(C, arch=arch, B=f'EqAlign without {F}', A='EqAlign', metric=m)
            r = find(C, arch=arch, B=f'EqAlign without {F}', A=f'Augmentation without {F}', metric=f'incons_rs_{F}')
            ok = bool(x) and x['ci_lo'] > 0 and full(x)
            npass += ok
            L.append(f'- **C9 [{F}]** EqAlign minus Augmentation, both trained without {F}, Top-1 on {F}: ' +
                     ('**INCOMPLETE**' if not x else f"{x['diff_B_minus_A']:+.1f} [{x['ci_lo']:.1f}, {x['ci_hi']:.1f}], "
                      f"{x['designs_better']}, sign p={x['p_sign']:.3f} -> **{'PASS' if ok else 'FAIL'}**"))
            if g:
                L.append(f"  - C9b gap, EqAlign without {F} minus EqAlign with {F}: {g['diff_B_minus_A']:+.1f} [{g['ci_lo']:.1f}, {g['ci_hi']:.1f}]")
            if r:
                L.append(f"  - restructured inconsistency on {F}, EqAlign minus Augmentation (both without {F}): {r['diff_B_minus_A']:+.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
        L.append(f'- Summary: C9 holds for {npass}/{len(E12_FAMS)} held-out families. If all hold, the paper\'s held-out '
                 'failure is specific to re-encodings of a different kind (H2); if some fail, it extends to unseen synthesis families.')
    return L


# ------------------------------------------------------------------------------------------------ io
def write_csv(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def md_table(rows, cols):
    head = '| config | n | div | ' + ' | '.join(cols) + ' |\n|' + '---|' * (len(cols) + 3) + '\n'
    return head + ''.join(f"| {r['arch']}: {r['config']} | {r['n_seeds']} | {r['n_diverged']} | " +
                          ' | '.join(fmt(r[f'{c}_mean'], r[f'{c}_sd'], c) for c in cols) + ' |\n' for r in rows)


def noise_floors(exp):
    out = {}
    for e in EXPDIRS.get(exp, [exp]):
        for f in glob.glob(f'runs/{e}/*/noise/noise.json'):
            out[f.split('/')[2]] = json.load(open(f))['noise_floor']
    return out


def sweep_report():
    rows, groups = [], {}
    for f in glob.glob('runs/E1*/*/*/sweep.json'):
        c = json.load(open(os.path.join(os.path.dirname(f), 'train.json')))['config']
        groups.setdefault((f"{c['arch']}_d{c['d']}_r{c['rounds']}", c['method']), []).append(f)
    for (arch, m), files in sorted(groups.items()):
        for bits in (4, 8, 16, 32, 64):
            for pol in ('level', 'emb', 'oracle'):
                per = []
                for f in files:
                    X = json.load(open(f)); num = den = 0.0
                    for key in sorted(set((x['circ'], x['kind']) for x in X)):
                        sel = [x for x in X if (x['circ'], x['kind']) == key and x['bits'] == bits]
                        topo = [x['wasted'] for x in sel if x['policy'] == 'topo']; vv = [x['wasted'] for x in sel if x['policy'] == pol]
                        if topo and vv:
                            den += np.mean(topo); num += np.mean(vv)
                    per.append(100 * (1 - num / den) if den else np.nan)
                per = np.array(per)
                rows.append(dict(arch=arch, regime=NAMES[m], bits=bits, policy=pol, n_seeds=len(per),
                                 fewer_refuted_pct_mean=float(np.nanmean(per)),
                                 se=float(np.nanstd(per, ddof=1) / np.sqrt(len(per))) if len(per) > 1 else 0.0))
    os.makedirs('results/E5', exist_ok=True); write_csv(rows, 'results/E5/sweep.csv'); print(f'results/E5/sweep.csv: {len(rows)} rows')


def main_e13():
    os.makedirs('results/E13', exist_ok=True)
    G = collect('E1a')
    G = {k: g for k, g in G.items() if g['files']['large']}
    if not G:
        print('E13: no raw_large.pkl yet (run tools.eval_large)'); return
    S = summarize(G, 'E13', 'large'); write_csv(S, 'results/E13/summary.csv')
    C = [r for r in comparisons(G, 'E13') if r['suite'] == 'large']; write_csv(C, 'results/E13/comparisons.csv')
    L = claims(S, C, 'E13 (large circuits, test only; design-level tests over 4 designs)', {}, suite='large')
    open('results/E13/claims.md', 'w').write('\n'.join(L) + '\n')
    cols = ['MAE', 'rho_syn', 'rho_inj', 'rho_H2', 'dMAE_H2', 'incons_rs', 'top1_seen', 'top1_ho', 'top1_H2', 'auroc_lvl']
    open('results/E13/report.md', 'w').write('# E13 report (large circuits)\n\n' + md_table(S, cols) + '\n')
    print(f'results/E13/: summary.csv ({len(S)} rows), comparisons.csv ({len(C)}), claims.md, report.md')


def main(exp):
    if exp == 'E5':
        return sweep_report()
    if exp == 'E13':
        return main_e13()
    os.makedirs(f'results/{exp}', exist_ok=True)
    G = collect(exp)
    if not G:
        print(f'{exp}: no finished runs yet'); return
    S = summarize(G, exp, 'epfl') + summarize(G, exp, 'iscas')
    write_csv(S, f'results/{exp}/summary.csv')
    ref = collect('E1a') if exp in ('E8', 'E9', 'E12') else None
    C = comparisons(G, exp, ref)
    write_csv(C, f'results/{exp}/comparisons.csv')
    T = trends(G, exp) if exp in ('E1', 'E2') else []
    write_csv(T, f'results/{exp}/trends.csv')
    noise = noise_floors(exp)
    if exp in ('E3', 'E10'):
        L = claims_e3(S, C, exp)
    elif exp in ('E8', 'E9'):
        L = claims_e8_e9(C, exp)
    elif exp == 'E12':
        L = claims_e12(C)
    else:
        L = claims(S, C, exp, noise)
    open(f'results/{exp}/claims.md', 'w').write('\n'.join(L) + '\n')
    cols = ['MAE', 'rho_syn', 'rho_inj', 'rho_H2', 'dMAE_H2', 'incons_rs', 'top1_seen', 'top1_all', 'top1_ho', 'top1_H2', 'auroc_lvl', 'worst_ratio']
    E = [r for r in S if r['suite'] == 'epfl']; I = [r for r in S if r['suite'] == 'iscas']
    rep = [f'# {exp} report\n', '## EPFL test (mean ± s.d. over seeds)\n', md_table(E, cols), '\n## ISCAS85/89\n',
           md_table(I, [c for c in cols if c != 'worst_ratio'])]
    if noise:
        rep.append('\n## Baseline seed-to-seed noise floor\n' + '\n'.join(f'- {k}: {v:.4f}' for k, v in sorted(noise.items())))
    open(f'results/{exp}/report.md', 'w').write('\n'.join(rep) + '\n')
    print(f'results/{exp}/: summary.csv ({len(S)} rows), comparisons.csv ({len(C)}), trends.csv ({len(T)}), claims.md, report.md')


if __name__ == '__main__':
    for e in (['E0', 'E3', 'E1', 'E8', 'E9', 'E12', 'E13', 'E4', 'E2', 'E10', 'E5'] if sys.argv[1] == 'all' else sys.argv[1:]):
        try:
            main(e)
        except Exception as ex:  # noqa: BLE001  one experiment's problem must not hide the others
            print(f'{e}: aggregation error {type(ex).__name__}: {ex}')
            if len(sys.argv[1:]) == 1 and sys.argv[1] != 'all':
                raise
