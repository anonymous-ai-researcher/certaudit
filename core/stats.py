"""Statistics with exactly the paper's definitions (analysis.py / make_numbers2.py), plus seed aggregation,
design-level cluster-bootstrap CIs, sign tests, Holm correction and scale-trend tests."""
import numpy as np
from scipy import stats as st

SYN = ['S1', 'S2', 'S3', 'S4']
INJ = ["('inj', 0.1)", "('inj', 0.3)", "('inj', 0.5)"]


def group_metrics(R, kinds=None, heldout=None):
    keys = ['mae_m_orig', 'mae_m_var', 'incons_rs', 'err_rs', 'err_rs_var', 'top1_rs', 'top1', 'auroc',
            'auroc_hard', 'chance', 'incons', 'frac_matched', 'incons_perm']
    out = {k: [] for k in keys}
    for r in R:
        if heldout:
            xs = [r['heldout'][heldout]] if 'heldout' in r else []
        else:
            xs = [r['kinds'][k] for k in kinds]
        for k in keys:
            v = [x[k] for x in xs if k in x]
            out[k].append(np.mean(v) if v else np.nan)
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def group_summary(R, kinds=None, heldout=None):
    c = group_metrics(R, kinds, heldout)
    s = {k: float(np.nanmean(v)) if np.isfinite(v).any() else float('nan') for k, v in c.items()}
    s['rho'] = float(np.nanmean(c['incons_rs']) / np.nanmean(c['err_rs']))
    s['mae_m_gap'] = s['mae_m_var'] - s['mae_m_orig']
    if np.isfinite(c['incons_perm']).any():
        s['gap_closed'] = 1 - float(np.nanmean(c['incons_rs']) / np.nanmean(c['incons_perm']))
    return s, c


def adv_matched(R):
    worst, orig = [], []
    for r in R:
        vs = [(x['mae_m_var'], x['mae_m_orig']) for x in r['adv'] if 'mae_m_var' in x]
        if vs:
            w = max(vs, key=lambda t: t[0] - t[1]); worst.append(w[0]); orig.append(w[1])
    if not worst:
        return dict(worst=np.nan, orig=np.nan, ratio=np.nan)
    worst, orig = np.array(worst), np.array(orig)
    return dict(worst=float(worst.mean()), orig=float(orig.mean()), ratio=float(worst.mean() / orig.mean()))


def _ho(R, key):
    v = [group_summary(R, heldout=h)[0][key] for h in ('H1', 'H2') if any('heldout' in r for r in R)]
    return float(np.mean(v)) if v else float('nan')


def _ho1(R, h, key):
    return group_summary(R, heldout=h)[0][key] if any('heldout' in r for r in R) else float('nan')


def _rho_all(R, kinds):
    c = group_metrics(R, kinds=kinds); return float(np.nanmean(c['incons']) / np.nanmean(c['mae_m_orig']))


G = lambda kinds, key: (lambda R: group_summary(R, kinds=kinds)[0].get(key, float('nan')))
# name -> (function of one seed's raw records, scale for reporting); names follow the paper's Table I
METRICS = {
    'MAE': (lambda R: float(np.mean([r['mae_orig'] for r in R])), 1),
    'dMAE_syn': (G(SYN, 'mae_m_gap'), 1), 'dMAE_inj': (G(INJ, 'mae_m_gap'), 1),
    'dMAE_ho': (lambda R: _ho(R, 'mae_m_gap'), 1),
    'dMAE_H1': (lambda R: _ho1(R, 'H1', 'mae_m_gap'), 1), 'dMAE_H2': (lambda R: _ho1(R, 'H2', 'mae_m_gap'), 1),
    'rho_syn': (G(SYN, 'rho'), 1), 'rho_inj': (G(INJ, 'rho'), 1), 'rho_ho': (lambda R: _ho(R, 'rho'), 1),
    'rho_H1': (lambda R: _ho1(R, 'H1', 'rho'), 1), 'rho_H2': (lambda R: _ho1(R, 'H2', 'rho'), 1),
    'rho_all': (G(SYN + INJ, 'rho'), 1),
    'gap_syn': (G(SYN, 'gap_closed'), 100), 'gap_inj': (G(INJ, 'gap_closed'), 100),
    'incons_rs': (G(SYN + INJ, 'incons_rs'), 1), 'err_rs': (G(SYN + INJ, 'err_rs'), 1),
    'incons_rs_syn': (G(SYN, 'incons_rs'), 1), 'incons_rs_inj': (G(INJ, 'incons_rs'), 1),
    'top1_seen': (G(SYN + INJ, 'top1_rs'), 100), 'top1_syn': (G(SYN, 'top1_rs'), 100),
    'top1_inj': (G(INJ, 'top1_rs'), 100), 'top1_ho': (lambda R: _ho(R, 'top1_rs'), 100),
    'top1_H1': (lambda R: _ho1(R, 'H1', 'top1_rs'), 100), 'top1_H2': (lambda R: _ho1(R, 'H2', 'top1_rs'), 100),
    'auroc_lvl': (G(SYN + INJ, 'auroc_hard'), 1), 'chance': (G(SYN + INJ, 'chance'), 100),
    # all-pair metrics (the paper's choice for the bidirectional encoder, whose state also sees fanout):
    # Top-1 over all proven pairs and rho = inconsistency over all pairs / error on the matched original gates
    'top1_all': (G(SYN + INJ, 'top1'), 100),
    'rho_allpairs_syn': (lambda R: _rho_all(R, SYN), 1), 'rho_allpairs_inj': (lambda R: _rho_all(R, INJ), 1),
    'worst_mae': (lambda R: adv_matched(R)['worst'], 1), 'worst_ratio': (lambda R: adv_matched(R)['ratio'], 1),
}


FAM_LAB = {'S1': 'S1', 'S2': 'S2', 'S3': 'S3', 'S4': 'S4', "('inj', 0.1)": 'I1', "('inj', 0.3)": 'I3', "('inj', 0.5)": 'I5'}
for _k, _l in FAM_LAB.items():                       # per-family breakdown
    METRICS[f'top1_{_l}'] = (G([_k], 'top1_rs'), 100)
    METRICS[f'rho_{_l}'] = (G([_k], 'rho'), 1)
    METRICS[f'dMAE_{_l}'] = (G([_k], 'mae_m_gap'), 1)


def seed_values(raws, name):
    f, sc = METRICS[name]
    return np.array([sc * f(R) for R in raws], dtype=float)


def mean_sd_ci(v):
    v = v[np.isfinite(v)]; n = len(v)
    if n == 0:
        return dict(mean=np.nan, sd=np.nan, n=0, ci_lo=np.nan, ci_hi=np.nan)
    m = float(v.mean()); sd = float(v.std(ddof=1)) if n > 1 else 0.0
    h = float(st.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return dict(mean=m, sd=sd, n=n, ci_lo=m - h, ci_hi=m + h)


# ---------------------------------------------------------------- design-level comparison (the paper's test)
def design_means(R, vec):
    src = np.array([r['src'] for r in R]); out = {}
    for s in sorted(set(src)):
        m = src == s; out[s] = float(np.nanmean(vec[m]))
    return out


PER_CIRCUIT = {
    'top1_seen': (lambda R: group_metrics(R, kinds=SYN + INJ)['top1_rs'], 100, False),
    'top1_ho': (lambda R: np.nanmean(np.vstack([group_metrics(R, heldout=h)['top1_rs'] for h in ('H1', 'H2')]), 0), 100, False),
    'incons_rs': (lambda R: group_metrics(R, kinds=SYN + INJ)['incons_rs'], 1, True),
    'incons_rs_inj': (lambda R: group_metrics(R, kinds=INJ)['incons_rs'], 1, True),
    'MAE': (lambda R: np.array([r['mae_orig'] for r in R]), 1, True),
    'auroc_lvl': (lambda R: group_metrics(R, kinds=SYN + INJ)['auroc_hard'], 1, False),
    'top1_all': (lambda R: group_metrics(R, kinds=SYN + INJ)['top1'], 100, False),
    'top1_H2': (lambda R: group_metrics(R, heldout='H2')['top1_rs'], 100, False),
    'dMAE_H2': (lambda R: group_metrics(R, heldout='H2')['mae_m_var'] - group_metrics(R, heldout='H2')['mae_m_orig'], 1, True),
}


# E12 per-family entries (S1..S4 and the three injection rates together): restructured Top-1, restructured
# inconsistency and matched-gate error increase of the held-out family
for _F, _K in [(f, [f]) for f in SYN] + [('inj', INJ)]:
    PER_CIRCUIT[f'top1_{_F}'] = ((lambda K: lambda R: group_metrics(R, kinds=K)['top1_rs'])(_K), 100, False)
    PER_CIRCUIT[f'incons_rs_{_F}'] = ((lambda K: lambda R: group_metrics(R, kinds=K)['incons_rs'])(_K), 1, True)
    PER_CIRCUIT[f'dMAE_{_F}'] = ((lambda K: lambda R: group_metrics(R, kinds=K)['mae_m_var'] - group_metrics(R, kinds=K)['mae_m_orig'])(_K), 1, True)
    if f'incons_rs_{_F}' not in METRICS:
        METRICS[f'incons_rs_{_F}'] = (G(_K, 'incons_rs'), 1)


def design_test(Ra_list, Rb_list, name, B=10000, seed=0):
    """B minus A, per design (cones and seeds pooled), 95% cluster-bootstrap CI over designs, two-sided sign
    test, and Wilcoxon signed-rank over designs. 'better' counts designs where B is better than A."""
    getter, sc, lower = PER_CIRCUIT[name]
    da = [design_means(R, getter(R)) for R in Ra_list]; db = [design_means(R, getter(R)) for R in Rb_list]
    designs = sorted(da[0])
    a = np.array([np.mean([d[s] for d in da]) for s in designs]); b = np.array([np.mean([d[s] for d in db]) for s in designs])
    # a design with no restructured pair of this family (adder under S3) has no value: drop it, and say so
    keep = np.isfinite(a) & np.isfinite(b); dropped = [s for s, k in zip(designs, keep) if not k]
    designs = [s for s, k in zip(designs, keep) if k]; a, b = a[keep], b[keep]
    diff = sc * (b - a)
    rng = np.random.default_rng(seed)
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    npos = int((diff > 0).sum()); nz = int((diff != 0).sum())
    p_sign = float(st.binomtest(npos, nz, 0.5).pvalue) if nz else 1.0
    try:
        p_wil = float(st.wilcoxon(diff).pvalue)
    except ValueError:
        p_wil = float('nan')
    better = int((diff < 0).sum()) if lower else int((diff > 0).sum())
    # exact sign-flip permutation test over designs (2^n flips; n = 7 gives 128) and a t-interval over designs
    n = len(diff)
    if n <= 16:
        import itertools
        flips = np.array(list(itertools.product([1, -1], repeat=n)))
        null = np.abs((flips * diff).mean(1)); p_perm = float((null >= abs(diff.mean()) - 1e-12).mean())
    else:
        p_perm = float('nan')
    sd = diff.std(ddof=1) if n > 1 else np.nan; h = st.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return dict(diff=float(diff.mean()), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                t_lo=float(diff.mean() - h), t_hi=float(diff.mean() + h),
                n_designs=n, n_better=better, p_sign=p_sign, p_wilcoxon=p_wil, p_perm=p_perm,
                per_design=dict(zip(designs, diff.round(6).tolist())), dropped_designs=dropped)


def welch(va, vb):
    va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
    if len(va) < 2 or len(vb) < 2:
        return float('nan')
    return float(st.ttest_ind(vb, va, equal_var=False).pvalue)


def holm(ps):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    ps = np.asarray(ps, dtype=float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def trend(xs, ys, B=5000, seed=0):
    """Slope of y on log2(x) with seed-level points, 95% bootstrap CI (resampling seeds within each x), and
    Spearman correlation. xs, ys: lists of equal length (one entry per seed-run)."""
    xs = np.log2(np.asarray(xs, float)); ys = np.asarray(ys, float); ok = np.isfinite(ys); xs, ys = xs[ok], ys[ok]
    if len(set(xs)) < 2:
        return dict(slope_per_doubling=np.nan, ci_lo=np.nan, ci_hi=np.nan, spearman=np.nan, n_widths=len(set(xs)))
    slope = float(np.polyfit(xs, ys, 1)[0]); rng = np.random.default_rng(seed); groups = {}
    for x, y in zip(xs, ys):
        groups.setdefault(x, []).append(y)
    bs = []
    for _ in range(B):
        X, Y = [], []
        for x, g in groups.items():
            g = np.asarray(g); s = g[rng.integers(0, len(g), len(g))]; X += [x] * len(s); Y += s.tolist()
        bs.append(np.polyfit(X, Y, 1)[0])
    mx = [np.mean(groups[x]) for x in sorted(groups)]           # Spearman on width means: descriptive (n = widths)
    rho = st.spearmanr(sorted(groups), mx)[0] if len(groups) > 2 else np.nan
    return dict(slope_per_doubling=slope, ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                spearman_means=float(rho), n_widths=len(groups))
