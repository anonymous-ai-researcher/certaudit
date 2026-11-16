"""Aggregate raw per-circuit evaluations into seed-level summaries, statistics and LaTeX macros."""
import pickle, json, glob, os, numpy as np
from scipy.stats import wilcoxon

FIXED = ['S1', 'S2', 'S3', 'S4', "('inj', 0.1)", "('inj', 0.3)", "('inj', 0.5)"]
KLAB = {'S1': 'S1', 'S2': 'S2', 'S3': 'S3', 'S4': 'S4', "('inj', 0.1)": 'I0.1', "('inj', 0.3)": 'I0.3', "('inj', 0.5)": 'I0.5'}


def load(tag):
    return pickle.load(open(f'raw_{tag}.pkl', 'rb'))


def circ_metrics(R):
    """Per-circuit metric vectors (pooled over the 7 fixed kinds)."""
    out = {k: [] for k in ['mae_orig', 'mae_var', 'incons_rs', 'err_rs', 'top1_rs', 'auroc', 'adv_ratio', 'adv_worst', 'incons', 'top1']}
    for r in R:
        out['mae_orig'].append(r['mae_orig'])
        ks = [r['kinds'][k] for k in FIXED]
        out['mae_var'].append(np.mean([x['mae'] for x in ks]))
        f = lambda key: np.mean([x[key] for x in ks if key in x]) if any(key in x for x in ks) else np.nan
        for key in ['incons_rs', 'err_rs', 'top1_rs', 'auroc', 'incons', 'top1']:
            out[key].append(f(key))
        adv = [x['mae'] for x in r['adv']]
        out['adv_ratio'].append(max(adv) / r['mae_orig'] if adv else np.nan)
        out['adv_worst'].append(max(adv) if adv else np.nan)
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def summary(R):
    c = circ_metrics(R)
    s = {k: float(np.nanmean(v)) for k, v in c.items()}
    s['rho'] = float(np.nanmean(c['incons_rs']) / np.nanmean(c['err_rs']))
    # ratio of means (robust to circuits with tiny original error); per-circuit ratios are shown as an ECDF
    s['adv_worst_ratio'] = float(np.nanmean(c['adv_worst']) / np.nanmean(c['mae_orig']))
    s['adv_ratio_median'] = float(np.nanmedian(c['adv_ratio']))
    # per kind
    s['by_kind'] = {}
    for k in FIXED:
        xs = [r['kinds'][k] for r in R]
        g = lambda key: float(np.nanmean([x.get(key, np.nan) for x in xs]))
        s['by_kind'][KLAB[k]] = dict(mae=g('mae'), top1_rs=g('top1_rs'), top1=g('top1'), auroc=g('auroc'),
                                    rho=g('incons_rs') / g('err_rs'), incons_rs=g('incons_rs'))
    return s, c


def seeds(prefix, n=3):
    tags = [f'{prefix}_s{i}' for i in range(n) if os.path.exists(f'raw_{prefix}_s{i}.pkl')]
    S = [summary(load(t)) for t in tags]
    return tags, S


def mstd(S, key):
    v = np.array([s[0][key] for s in S]); return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)


def paired_test(Ca, Cb, key):
    a, b = Ca[key], Cb[key]; m = ~(np.isnan(a) | np.isnan(b))
    try:
        return float(wilcoxon(a[m], b[m]).pvalue)
    except ValueError:
        return float('nan')


def bootstrap_ci(x, B=2000, seed=0):
    x = x[~np.isnan(x)]; rng = np.random.default_rng(seed)
    bs = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ----------------------------------------------------------------------------- revised metrics
SYN = ['S1', 'S2', 'S3', 'S4']
INJ = ["('inj', 0.1)", "('inj', 0.3)", "('inj', 0.5)"]


def group_metrics(R, kinds=None, heldout=None):
    """Pooled per-circuit means over a group of variant families. Returns dict of per-circuit arrays."""
    keys = ['mae_m_orig', 'mae_m_var', 'incons_rs', 'err_rs', 'err_rs_var', 'top1_rs', 'top1', 'auroc',
            'auroc_hard', 'chance', 'incons', 'frac_matched']
    out = {k: [] for k in keys}
    for r in R:
        if heldout:
            if 'heldout' not in r:
                xs = []
            else:
                xs = [r['heldout'][heldout]]
        else:
            xs = [r['kinds'][k] for k in kinds]
        for k in keys:
            v = [x[k] for x in xs if k in x]
            out[k].append(np.mean(v) if v else np.nan)
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def group_summary(R, kinds=None, heldout=None):
    c = group_metrics(R, kinds, heldout)
    s = {k: float(np.nanmean(v)) for k, v in c.items()}
    s['rho'] = float(np.nanmean(c['incons_rs']) / np.nanmean(c['err_rs']))
    s['mae_m_gap'] = s['mae_m_var'] - s['mae_m_orig']
    return s, c


def adv_matched(R):
    """Worst case over 16 certified rewrites on MATCHED gates: per circuit, max over variants of the variant
    error on matched gates, and the original error on the same gates for that variant."""
    worst, orig = [], []
    for r in R:
        vs = [(x['mae_m_var'], x['mae_m_orig']) for x in r['adv'] if 'mae_m_var' in x]
        if vs:
            w = max(vs, key=lambda t: t[0] - t[1])   # largest degradation
            worst.append(w[0]); orig.append(w[1])
    worst, orig = np.array(worst), np.array(orig)
    return dict(worst=float(worst.mean()), orig=float(orig.mean()), ratio=float(worst.mean() / orig.mean()),
                worst_c=worst, orig_c=orig)


def design_means(R, vec):
    src = np.array([r['src'] for r in R]); out = {}
    for s in sorted(set(src)):
        m = src == s; out[s] = float(np.nanmean(vec[m]))
    return out


def design_test(Ra_list, Rb_list, getter, B=5000, seed=0):
    """Design-level comparison pooled over seeds: per-design mean (over cones and seeds) of getter(R) for A and B.
    Returns mean difference (B - A), cluster-bootstrap 95% CI over designs, and sign-test p (two-sided)."""
    from scipy.stats import binomtest
    da = [design_means(R, getter(R)) for R in Ra_list]; db = [design_means(R, getter(R)) for R in Rb_list]
    designs = sorted(da[0])
    a = np.array([np.mean([d[s] for d in da]) for s in designs]); b = np.array([np.mean([d[s] for d in db]) for s in designs])
    diff = b - a
    rng = np.random.default_rng(seed)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
    npos = int((diff > 0).sum()); nz = int((diff != 0).sum())
    p = binomtest(npos, nz, 0.5).pvalue if nz else 1.0
    return dict(diff=float(diff.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                n_pos=npos, n=len(diff), p_sign=float(p), designs=designs, a=a.tolist(), b=b.tolist())
