"""E17 aggregation: the audit and the repairs under certified ABC rewrites (data/abc.pkl, test only).

  python -m tools.aggregate_e17            -> results/E17/{summary.csv, comparisons.csv, claims.md, report.md}

Per model configuration (seed means): for every ABC family and pooled over the four, the paper's metrics
(rho = restructured inconsistency / restructured error, restructured Top-1, matched-gate error increase dMAE,
restructured inconsistency), next to the same models' mockturtle synthesis families S1-S4 from raw_epfl.pkl.
Design-level tests (7 EPFL designs, cones and seeds pooled; ISCAS 17 designs) exactly as tools/aggregate.py.
Claims E17-C1..C4 below mirror the paper's C1-C3 with the ABC families in place of S1-S4; rules fixed in advance."""
import csv, glob, json, os, numpy as np
from core.stats import METRICS, PER_CIRCUIT, G as GM, group_metrics, seed_values, mean_sd_ci, design_test, welch, holm
from tools.aggregate import raw, collect, label, NAMES, OTHERS

ABC = ['A1_resyn2', 'A2_dc2', 'A3_compress2rs', 'A4_lutmap']
SHORT = {'A1_resyn2': 'resyn2', 'A2_dc2': 'dc2', 'A3_compress2rs': 'compress2rs', 'A4_lutmap': 'lutmap'}
GROUPS = {'abc': ABC, 'abc_rw': ABC[:3], **{SHORT[f]: [f] for f in ABC}}
for g, K in GROUPS.items():
    METRICS[f'rho_{g}'] = (GM(K, 'rho'), 1); METRICS[f'top1_{g}'] = (GM(K, 'top1_rs'), 100)
    METRICS[f'top1all_{g}'] = (GM(K, 'top1'), 100); METRICS[f'dMAE_{g}'] = (GM(K, 'mae_m_gap'), 1)
    METRICS[f'incons_rs_{g}'] = (GM(K, 'incons_rs'), 1); METRICS[f'frac_rs_{g}'] = (GM(K, 'frac_rs'), 100)
    METRICS[f'mae_var_{g}'] = (GM(K, 'mae_m_var'), 1)
    PER_CIRCUIT[f'top1_{g}'] = ((lambda K: lambda R: group_metrics(R, kinds=K)['top1_rs'])(K), 100, False)
    PER_CIRCUIT[f'incons_rs_{g}'] = ((lambda K: lambda R: group_metrics(R, kinds=K)['incons_rs'])(K), 1, True)
    PER_CIRCUIT[f'dMAE_{g}'] = ((lambda K: lambda R: group_metrics(R, kinds=K)['mae_m_var'] - group_metrics(R, kinds=K)['mae_m_orig'])(K), 1, True)
# group_metrics does not carry frac_rs; add it (fraction of proven pairs that are restructured)
import core.stats as _st
_gm = _st.group_metrics


def _gm2(R, kinds=None, heldout=None):
    out = _gm(R, kinds, heldout)
    if kinds:
        out['frac_rs'] = np.array([np.mean([r['kinds'][k]['frac_rs'] for k in kinds if 'frac_rs' in r['kinds'].get(k, {})])
                                   if any('frac_rs' in r['kinds'].get(k, {}) for k in kinds) else np.nan for r in R])
    return out


_st.group_metrics = _gm2
import core.stats as S2  # noqa: E402
S2.group_summary.__globals__['group_metrics'] = _gm2

COLS = (['rho_syn', 'top1_syn'] + [f'{m}_{g}' for g in ('abc', 'abc_rw', 'lutmap') for m in ('rho', 'top1', 'dMAE', 'incons_rs')] +
        [f'top1_{SHORT[f]}' for f in ABC] + [f'rho_{SHORT[f]}' for f in ABC] + ['top1all_abc', 'frac_rs_abc', 'mae_var_abc'])
EXPS = os.environ.get('E17_EXPS', 'E1a,E2,E15,E3,E10').split(',')


def configs():
    """config -> files per suite, from the same collection as tools/aggregate.py (primary evaluation only)."""
    out = {}
    for e in EXPS:
        for k, g in collect(e).items():
            if k[6] not in ('z', 'shared') or k[5].startswith(('probe', 'holdout', 'h2')):
                continue
            runs = sorted(set(g['runs']))
            f = {'epfl': [], 'abc': [], 'abc_iscas': []}
            for r in runs:
                if os.path.exists(f'{r}/raw_abc.pkl'):
                    f['abc'].append(f'{r}/raw_abc.pkl'); f['epfl'].append(f'{r}/raw_epfl.pkl')
                    if os.path.exists(f'{r}/raw_abc_iscas.pkl'):
                        f['abc_iscas'].append(f'{r}/raw_abc_iscas.pkl')
            if f['abc']:
                out[(e,) + k] = dict(files=f, n_div=g['n_diverged'])
    return out


def row(key, g, suite):
    e, arch, m = key[0], key[1], key[2]
    R = [raw(p) for p in g['files'][suite]]
    r = dict(exp=e, suite=suite, arch=arch, method=m, config=label(key[1:]), n_seeds=len(R), n_diverged=g['n_div'])
    for c in COLS:
        if c in ('rho_syn', 'top1_syn'):
            if suite != 'abc':
                continue
            v = seed_values([raw(p) for p in g['files']['epfl']], c)
        else:
            v = seed_values(R, c)
        s = mean_sd_ci(v); r[f'{c}_mean'], r[f'{c}_sd'], r[f'{c}_ci_lo'], r[f'{c}_ci_hi'] = s['mean'], s['sd'], s['ci_lo'], s['ci_hi']
    return r


def cmp(gB, gA, nB, nA, ctx, suite, mets):
    Ra = [raw(p) for p in gA['files'][suite]]; Rb = [raw(p) for p in gB['files'][suite]]
    if not Ra or not Rb:
        return []
    out = []
    for met in mets:
        d = design_test(Ra, Rb, met)
        out.append(dict(ctx, suite=suite, B=nB, A=nA, metric=met, diff_B_minus_A=d['diff'], ci_lo=d['ci_lo'], ci_hi=d['ci_hi'],
                        t_lo=d['t_lo'], t_hi=d['t_hi'], designs_better=f"{d['n_better']}/{d['n_designs']}",
                        dropped_designs=';'.join(d.get('dropped_designs', [])), p_sign=d['p_sign'], p_perm=d['p_perm'],
                        p_welch_seeds=welch(seed_values(Ra, met), seed_values(Rb, met)), n_seeds_A=len(Ra), n_seeds_B=len(Rb)))
    return out


def comparisons(C):
    rows = []
    mets = ['top1_abc', 'top1_abc_rw', 'top1_lutmap', 'incons_rs_abc', 'incons_rs_abc_rw', 'incons_rs_lutmap', 'dMAE_abc']
    for suite in ('abc', 'abc_iscas'):
        for (e, arch) in sorted(set((k[0], k[1]) for k in C)):
            M = {k[2]: g for k, g in C.items() if k[0] == e and k[1] == arch and k[5] == 1 and k[6] in ('', 'train_pi=shared')}
            ctx = dict(exp=e, arch=arch)
            if 'eqalign_ph' in M:
                for o in OTHERS:
                    if o in M:
                        rows += cmp(M['eqalign_ph'], M[o], 'EqAlign', NAMES[o], ctx, suite, mets)
            if 'aug' in M and 'base' in M:
                rows += cmp(M['aug'], M['base'], 'Augmentation', 'Baseline', ctx, suite, mets)
            if arch == 'deepgate2':
                for k, g in C.items():
                    if k[0] == e and k[2] == 'dg2_eqalign' and k[6] == 'train_pi=shared':
                        ctl = [g2 for k2, g2 in C.items() if k2[0] == e and k2[2] == 'dg2_control' and k2[4] == k[4] and k2[6] == k[6]]
                        rel = [g2 for k2, g2 in C.items() if k2[2] == 'dg2_released' and k2[6] == '']
                        if ctl:
                            rows += cmp(g, ctl[0], label(k[1:]), 'DG2 fine-tuned (control)', ctx, suite, mets)
                        if rel:
                            rows += cmp(g, rel[0], label(k[1:]), 'DeepGate2 released', ctx, suite, mets)
    groups = {}
    for i, r in enumerate(rows):
        if r['B'] == 'EqAlign':
            groups.setdefault((r['exp'], r['arch'], r['suite'], r['metric']), []).append(i)
    for idx in groups.values():
        for col in ('p_sign', 'p_perm', 'p_welch_seeds'):
            for i, q in zip(idx, holm([rows[i][col] if rows[i][col] == rows[i][col] else 1.0 for i in idx])):
                rows[i][col + '_holm'] = q
    return rows


def find(C, **f):
    m = [r for r in C if all(r.get(k) == v for k, v in f.items())]
    return m[0] if m else None


def claims(S, C):
    noise = {}
    for f in sorted(glob.glob('runs/*/*/noise/noise.json'), key=lambda f: f.startswith('runs/E1a/')):
        noise[f.split('/')[2]] = json.load(open(f))['noise_floor']
    L = ['# E17 claims check (ABC rewrite families; rules fixed before any model was evaluated on them)\n',
         'C1: rho_abc > 0.5 and restructured inconsistency / seed-noise floor > 2 (as the paper\'s C1). '
         'C2: Augmentation - Baseline restructured inconsistency on abc: bootstrap CI and t-interval below 0. '
         'C3: EqAlign - every other regime, restructured Top-1 on abc: CI above 0 and better on every design. '
         'C4 (DeepGate2): fine-tuned EqAlign - control and - released, Top-1 on abc: CI above 0. '
         'Each rule is also evaluated on abc_rw (resyn2, dc2, compress2rs) and on lutmap separately, since the LUT round '
         'trip is a different kind of rewrite; transfer across kinds is reported, never required.\n']
    for (e, arch) in sorted(set((r['exp'], r['arch']) for r in S if r['suite'] == 'abc')):
        R = {r['method']: r for r in S if r['exp'] == e and r['arch'] == arch and r['suite'] == 'abc'}
        L.append(f'\n## {e} {arch}\n')
        b = R.get('base')
        if b:
            for g in ('abc', 'abc_rw', 'lutmap'):
                nf = noise.get(arch); ratio = b[f'incons_rs_{g}_mean'] / nf if nf else float('nan')
                ok = b[f'rho_{g}_mean'] > 0.5 and (ratio > 2 if nf else True)
                L.append(f"- **C1 ({g})** baseline rho {b[f'rho_{g}_mean']:.2f} ± {b[f'rho_{g}_sd']:.2f} (mockturtle S1-S4: "
                         f"{b['rho_syn_mean']:.2f}), inconsistency / noise floor {ratio:.1f}x, Top-1 {b[f'top1_{g}_mean']:.1f}% "
                         f"(S1-S4: {b['top1_syn_mean']:.1f}%) -> **{'PASS' if ok else 'FAIL'}**" + ('' if nf else ' (no noise floor: rho part only)'))
            for g in ('abc', 'abc_rw', 'lutmap'):
                c = find(C, exp=e, arch=arch, suite='abc', B='Augmentation', A='Baseline', metric=f'incons_rs_{g}')
                if c:
                    ok = c['ci_hi'] < 0 and c['t_hi'] < 0
                    L.append(f"- **C2 ({g})** Aug - Base inconsistency {c['diff_B_minus_A']:+.4f} [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}], "
                             f"better on {c['designs_better']} -> **{'PASS' if ok else 'FAIL'}**")
        if 'eqalign_ph' in R:
            for g in ('abc', 'abc_rw', 'lutmap'):
                parts, fails = [], []
                for o in OTHERS:
                    c = find(C, exp=e, arch=arch, suite='abc', B='EqAlign', A=NAMES[o], metric=f'top1_{g}')
                    if not c:
                        continue
                    n, d = map(int, c['designs_better'].split('/'))
                    ok = c['ci_lo'] > 0 and n == d
                    parts.append(f"vs {NAMES[o]} {c['diff_B_minus_A']:+.1f} pts [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}], {c['designs_better']}")
                    fails += [] if ok else [NAMES[o]]
                if parts:
                    L.append(f"- **C3 ({g})** " + '; '.join(parts) + f" -> **{'PASS' if not fails else 'FAIL (' + ', '.join(fails) + ')'}**")
        if arch == 'deepgate2':
            for c in [x for x in C if x['exp'] == e and x['suite'] == 'abc' and x['metric'] in ('top1_abc', 'top1_abc_rw', 'top1_lutmap')
                      and x['A'] in ('DG2 fine-tuned (control)', 'DeepGate2 released')]:
                L.append(f"- **C4 ({c['metric'][5:]})** {c['B']} - {c['A']}: {c['diff_B_minus_A']:+.1f} pts [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}], "
                         f"{c['designs_better']} -> **{'PASS' if c['ci_lo'] > 0 else 'FAIL'}**")
    return L


def write_csv(rows, path):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    os.makedirs('results/E17', exist_ok=True)
    C0 = configs()
    if not C0:
        print('E17: no raw_abc.pkl yet'); return
    S = [row(k, g, s) for k, g in sorted(C0.items(), key=lambda kv: str(kv[0])) for s in ('abc', 'abc_iscas') if g['files'][s]]
    C = comparisons(C0)
    write_csv(S, 'results/E17/summary.csv'); write_csv(C, 'results/E17/comparisons.csv')
    open('results/E17/claims.md', 'w').write('\n'.join(claims(S, C)) + '\n')
    cols = ['rho_syn', 'rho_abc', 'rho_abc_rw', 'rho_lutmap', 'top1_syn', 'top1_abc', 'top1_abc_rw', 'top1_lutmap', 'dMAE_abc', 'frac_rs_abc']
    rep = ['# E17 report: certified ABC rewrites (mean ± s.d. over seeds)\n',
           'S1-S4 columns: the same models on the paper\'s mockturtle families, for comparison.\n']
    for suite in ('abc', 'abc_iscas'):
        rep.append(f'\n## {"EPFL" if suite == "abc" else "ISCAS85/89"}\n')
        rep.append('| exp | config | n | ' + ' | '.join(cols) + ' |\n|' + '---|' * (len(cols) + 3))
        for r in S:
            if r['suite'] == suite:
                rep.append(f"| {r['exp']} | {r['arch']}: {r['config']} | {r['n_seeds']} | " +
                           ' | '.join('' if f'{c}_mean' not in r else f"{r[f'{c}_mean']:.3f} ± {r[f'{c}_sd']:.3f}" if c.startswith(('rho', 'dMAE'))
                                      else f"{r[f'{c}_mean']:.1f} ± {r[f'{c}_sd']:.1f}" for c in cols) + ' |')
    open('results/E17/report.md', 'w').write('\n'.join(rep) + '\n')
    print(f'results/E17/: summary.csv ({len(S)} rows), comparisons.csv ({len(C)}), claims.md, report.md')


if __name__ == '__main__':
    main()
