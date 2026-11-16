"""Revised figures (after independent review). Same design system (figstyle.py):
fill color = model identity; outlined ink = non-learned procedure; one sans face; one y-axis per panel."""
import json, pickle, glob, os, sys, numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figstyle import PRINT_SCALE, pack_row, bar_style, EDGE, LW_BOOST, REF_EDGE, C, NAMES, MARK, HATCH, LS, INK, INK2, MUTED, RULE, GRID, COL_W, TXT_W, panel_label, save, badge_title, family_bands, end_label, callout
from analysis import load, group_summary, group_metrics, adv_matched, SYN, INJ

LAM = os.environ.get('LAMSTAR', '0.2'); TTL = os.environ.get('TTLAM', '0.1'); CL = os.environ.get('CIRCLAM', '0.2')
PREF = {'base': 'base', 'aug': 'aug', 'tt': f'ttsim_ph_l{TTL}', 'circ': f'circ_ph_l{CL}', 'ours': f'eqalign_ph_l{LAM}'}
METHODS = [m for m in ['base', 'aug', 'tt', 'circ', 'ours'] if glob.glob(f'raw_{PREF[m]}_s*.pkl')]
FAMS = [('S1', ['S1']), ('S2', ['S2']), ('S3', ['S3']), ('S4', ['S4']), ('I.1', ["('inj', 0.1)"]),
        ('I.3', ["('inj', 0.3)"]), ('I.5', ["('inj', 0.5)"]), ('H1', 'H1'), ('H2', 'H2')]
RAW = {m: [load(t[4:-4]) for t in sorted(glob.glob(f'raw_{PREF[m]}_s*.pkl'))] for m in METHODS}


def fam_stat(R, fam, key):
    lab, sel = fam
    s = group_summary(R, heldout=sel)[0] if isinstance(sel, str) else group_summary(R, kinds=sel)[0]
    return s[key]


def seeds_stat(m, fam, key):
    v = np.array([fam_stat(R, fam, key) for R in RAW[m]]); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0)


def offs(n, w):
    return (np.arange(n) - (n - 1) / 2) * w


def heldout_divider(ax, n_fixed=7):
    ax.axvline(n_fixed - 0.5, color=RULE, lw=0.6)
    ax.text(n_fixed + 0.5, 0.98, 'held-out', transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=6.0, color=INK2)


def capped_bars(ax, x, mu, sd, cap, color, w, hatch=None, m=None):
    """Bars clipped at `cap`; clipped bars get their value printed above the cap (no broken axis)."""
    mu = np.asarray(mu); sd = np.asarray(sd)
    shown = np.minimum(mu, cap)
    st = bar_style(m) if m else dict(facecolor=color, edgecolor='white', hatch=hatch, lw=0.35)
    ax.bar(x, shown, width=w, zorder=3, **st)
    for xi, m_, s_, sh in zip(x, mu, sd, shown):
        if m_ <= cap:
            ax.errorbar(xi, m_, yerr=s_, color=INK2, elinewidth=0.6, capsize=0, zorder=4)
    return [(xi, m_) for xi, m_ in zip(x, mu) if m_ > cap]


def clip_note(ax, clipped, cap, fmt):
    """One label per clipped family: the range of clipped values, printed above the cap."""
    if not clipped:
        return
    groups = {}
    for xi, v in clipped:
        groups.setdefault(int(round(xi)), []).append(v)
    for g, vs in groups.items():
        txt = fmt(min(vs)) if len(vs) == 1 or fmt(min(vs)) == fmt(max(vs)) else f'{fmt(min(vs))}–{fmt(max(vs))}'
        ax.text(g, cap * 1.03, txt, ha='center', va='bottom', fontsize=6.0, color=INK)


def fig_audit():
    fig, axs = plt.subplots(1, 4, figsize=(TXT_W / PRINT_SCALE, 1.88), gridspec_kw=dict(wspace=0.5, width_ratios=[1.45, 1.45, 0.95, 1.0]))
    nm = len(METHODS); w = 0.8 / nm
    # (a) degradation on matched gates
    ax = axs[0]
    clipped = []
    for i, m in enumerate(METHODS):
        mu, sd = zip(*[seeds_stat(m, f, 'mae_m_gap') for f in FAMS])
        clipped += capped_bars(ax, np.arange(len(FAMS)) + offs(nm, w)[i], mu, sd, 0.03, C[m], w * 0.92, HATCH.get(m), m=m)
    clip_note(ax, clipped, 0.03, lambda v: f'{v:.2f}')
    ax.axhline(0, color=INK, lw=0.6); ax.set_ylim(-0.008, 0.038)
    ax.set_xticks(range(len(FAMS))); ax.set_xticklabels([f[0] for f in FAMS])
    ax.set_ylabel('$\\Delta$MAE on matched gates')
    family_bands(ax, label_y=-0.135); badge_title(ax, 'a', 'Error on the same functions')
    # (b) rho by family, with seed-to-seed noise floor of the baseline
    ax = axs[1]
    clipped = []
    for i, m in enumerate(METHODS):
        mu, sd = zip(*[seeds_stat(m, f, 'rho') for f in FAMS])
        clipped += capped_bars(ax, np.arange(len(FAMS)) + offs(nm, w)[i], mu, sd, 2.2, C[m], w * 0.92, HATCH.get(m), m=m)
    clip_note(ax, clipped, 2.2, lambda v: f'{v:.1f}')
    ax.set_ylim(0, 2.7)
    if os.path.exists('noise_floor.json'):
        nf = json.load(open('noise_floor.json'))['base']
        err = np.mean([group_summary(R, kinds=SYN + INJ)[0]['err_rs'] for R in RAW['base']])
        ax.axhspan(0, nf / err, color='#9aa1a9', alpha=0.35, zorder=4, lw=0)
        ax.add_patch(plt.Rectangle((0.035, 0.885), 0.075, 0.045, transform=ax.transAxes, facecolor='#9aa1a9',
                                   alpha=0.35, lw=0, zorder=7, clip_on=False))
        ax.text(0.125, 0.907, 'seed-to-seed noise', transform=ax.transAxes, fontsize=6.0, color=INK,
                ha='left', va='center', zorder=7)
    ax.axhline(1.0, color=INK, lw=0.6)
    ax.set_xticks(range(len(FAMS))); ax.set_xticklabels([f[0] for f in FAMS])
    ax.set_ylabel('$\\rho$ = inconsistency / error')
    family_bands(ax, label_y=-0.135); badge_title(ax, 'b', 'Consistency ratio')
    # (c) worst case on matched gates: ECDF of per-circuit worst/orig (all seeds pooled)
    ax = axs[2]
    for m in METHODS:
        r = np.concatenate([adv_matched(R)['worst_c'] / adv_matched(R)['orig_c'] for R in RAW[m]])
        r = np.sort(r[np.isfinite(r)]); y = np.arange(1, len(r) + 1) / len(r)
        ax.step(r, y, where='post', color=C[m], lw=1.2 + LW_BOOST.get(m, 0), ls=LS.get(m, '-'))
    ax.axvline(1.0, color=INK, lw=0.6)
    ax.set_xscale('log'); ax.set_xlim(0.8, 12); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(['1', '2', '5', '10'])
    ax.minorticks_off(); ax.grid(axis='x', color=GRID, lw=0.5)
    ax.set_xlabel('worst / original MAE\n(same gates)'); ax.set_ylabel('fraction of circuits')
    med = {m: np.median(np.concatenate([adv_matched(R)['worst_c'] / adv_matched(R)['orig_c'] for R in RAW[m]])) for m in METHODS}
    ax.axhline(0.5, color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.plot([med['ours'], med['base']], [0.5, 0.5], color=INK, lw=0.8, zorder=5)
    for m in ('ours', 'base'):
        ax.plot(med[m], 0.5, marker=MARK[m], color=C[m], mec='white', mew=0.6, ms=4, zorder=6)
    ax.text(med['base'] * 1.1, 0.46, f"median\nBaseline {med['base']:.2f}×\nEqAlign {med['ours']:.2f}×", fontsize=6.0, color=INK, va='top', ha='left', linespacing=1.15)
    badge_title(ax, 'c', 'Worst of 16 rewrites')
    # (d) rho by depth (normalizes the error-depth confound)
    ax = axs[3]
    bins = [0, 4, 8, 12, 100]; labs = ['1–4', '5–8', '9–12', '>12']
    for m in METHODS:
        lev, dif, err = [], [], []
        for R in RAW[m]:
            for r in R:
                for k in SYN + INJ:
                    p = r['kinds'][k].get('_pairs')
                    if p is None:
                        continue
                    lev.append(p['lev'][p['rs']]); dif.append(p['diff'][p['rs']]); err.append(p['err'][p['rs']])
        lev, dif, err = map(np.concatenate, (lev, dif, err))
        idx = np.digitize(lev, bins[1:-1], right=True)
        rho = [dif[idx == b].mean() / err[idx == b].mean() for b in range(len(labs))]
        ax.plot(range(len(labs)), rho, color=C[m], marker=MARK[m], mfc='white', mec=EDGE.get(m, C[m]), mew=1.0, ms=3.2, lw=1.3 + LW_BOOST.get(m, 0), ls=LS.get(m, '-'), zorder=4 + (m == 'ours'))
        if m in ('base', 'ours'):
            end_label(ax, 1.9, rho[2], NAMES[m].split(' (')[0], C[m], dx=0, dy=7 if m == 'base' else -7, ha='center')
    ax.axhline(1.0, color=INK, lw=0.6)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=6.2)
    ax.set_xlabel('logic depth of gate'); ax.set_ylabel('$\\rho$ within depth bin'); ax.set_ylim(0.4, None)
    badge_title(ax, 'd', 'Depth, error-normalized')
    h = [Patch(label=NAMES[m], **bar_style(m)) for m in METHODS]
    fig.legend(handles=h, loc='upper center', ncol=len(METHODS), bbox_to_anchor=(0.5, 1.08), handlelength=1.4, columnspacing=1.6)
    pack_row(fig, list(axs), [1.45, 1.45, 0.95, 1.0])
    save(fig, 'fig_audit')


def fig_retrieval():
    """(a) per-family Top-1 retrieval; (b) recall@k from recall_at_k.json (computed by fig_recall)."""
    fig, axs = plt.subplots(1, 2, figsize=(COL_W / PRINT_SCALE, 1.75), gridspec_kw=dict(wspace=0.42, width_ratios=[1.55, 1]))
    ax = axs[0]; nm = len(METHODS); w = 0.86 / nm
    for i, m in enumerate(METHODS):
        mu, sd = zip(*[seeds_stat(m, f, 'top1_rs') for f in FAMS])
        ax.bar(np.arange(len(FAMS)) + offs(nm, w)[i], 100 * np.array(mu), width=w * 0.92, **bar_style(m, 0.3),
               yerr=100 * np.array(sd), error_kw=dict(elinewidth=0.5, ecolor=INK2), zorder=3)
    ch = np.mean([fam_stat(RAW['base'][0], f, 'chance') for f in FAMS]) * 100
    ax.axhline(ch, color=INK, lw=0.6, ls=(0, (2, 1.5)), zorder=5)
    ax.set_xticks(range(len(FAMS))); ax.set_xticklabels([f[0] for f in FAMS], fontsize=6.1)
    ax.set_ylabel('top-1 retrieval (%)'); ax.set_ylim(0, 100)
    family_bands(ax, groups=(('syn.', -0.5, 3.5, 'syn'), ('inj.', 3.5, 6.5, 'inj'), ('held-out', 6.5, 8.5, 'ho')), label_y=-0.13)
    badge_title(ax, 'a', 'Per rewrite family')
    ax = axs[1]
    rk = json.load(open('recall_at_k.json')); ks = np.array([1, 2, 3, 5, 10, 20, 50])
    for m in METHODS:
        if m in rk:
            ax.plot(ks, rk[m]['mean'], color=C[m], marker=MARK[m], mfc='white', mec=EDGE.get(m, C[m]), mew=0.9, ms=2.8, lw=1.2 + LW_BOOST.get(m, 0), ls=LS.get(m, '-'))
    ax.set_xscale('log'); ax.set_xticks([1, 5, 10, 50]); ax.set_xticklabels(['1', '5', '10', '50']); ax.minorticks_off()
    ax.set_xlim(0.8, 60); ax.set_ylim(0, 100); ax.grid(axis='x', color=GRID, lw=0.5)
    ax.set_xlabel('k (candidates)'); ax.set_ylabel('recall@k (%)')
    if 'ours' in rk and 'base' in rk:
        o1, b1 = rk['ours']['mean'][0], rk['base']['mean'][0]
        ax.annotate('', xy=(1, o1 - 1.5), xytext=(1, b1 + 1.5), arrowprops=dict(arrowstyle='<->', color=INK, lw=0.6, shrinkA=0, shrinkB=0))
        ax.text(1.08, 24.5, f'+{round(o1) - round(b1)} pts at k=1', fontsize=6.1, color=INK, fontweight='bold', va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'), zorder=8)
    badge_title(ax, 'b', 'As a candidate list')
    h = [Patch(label=NAMES[m], **bar_style(m)) for m in METHODS]
    fig.legend(handles=h, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.17), handlelength=1.2, columnspacing=1.0, fontsize=6)
    pack_row(fig, list(axs), [1.55, 1])
    save(fig, 'fig_retrieval')


def fig_recall():
    import torch
    from model import AIGEncoder, collate
    from evaluate import shash
    D = pickle.load(open('data.pkl', 'rb'))
    ks = np.array([1, 2, 3, 5, 10, 20, 50]); out = {}
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    for m in METHODS:
        curves = []
        for t in sorted(glob.glob(f'v_{PREF[m]}_s*.pt')):
            net = AIGEncoder(64, 2, proj='_ph' in t); net.load_state_dict(torch.load(t)); net.eval()
            ranks = []
            with torch.no_grad():
                for n, d in enumerate(D['test']):
                    zo = net.z(net(collate([d]))[0]).numpy(); sho = shash(d)
                    go = np.arange(1 + d['n_pi'], 1 + d['n_pi'] + len(d['fanins']))
                    Zo = zo[go] / (np.linalg.norm(zo[go], axis=1, keepdims=True) + 1e-9)
                    for v in D['tvars'][n].values():
                        keep = (v['P'] == 0) & (sho[v['I']] != shash(v['var'])[v['J']])
                        if not keep.any():
                            continue
                        zv = net.z(net(collate([v['var']]))[0]).numpy()[v['J'][keep]]
                        zv = zv / (np.linalg.norm(zv, axis=1, keepdims=True) + 1e-9)
                        sims = zv @ Zo.T; true = sims[np.arange(len(zv)), v['I'][keep] - go[0]]
                        ranks.append((sims >= true[:, None] - 1e-7).sum(1))   # pessimistic ties
            r = np.concatenate(ranks); curves.append([(r <= k).mean() * 100 for k in ks])
        cur = np.array(curves); mu = cur.mean(0); out[m] = dict(mean=mu.tolist(), sd=cur.std(0, ddof=1).tolist() if len(cur) > 1 else None)
    plt.close(fig); json.dump(out, open('recall_at_k.json', 'w'), indent=1)


def fig_sweep():
    rows = {m: [x for f in sorted(glob.glob(f'sweep2_v_{PREF[m]}_s*.json')) for x in json.load(open(f))] for m in METHODS}
    rows = {m: r for m, r in rows.items() if r}
    any_rows = next(iter(rows.values()))
    bitsL = sorted(set(x['bits'] for x in any_rows))

    from sweepagg import reduction
    files_of = {m: sorted(glob.glob(f'sweep2_v_{PREF[m]}_s*.json')) for m in rows}

    def rel(rr, pol, bits, field='wasted', add_emb=False, m=None):
        """Shared aggregation (sweepagg.reduction): pooled per seed, mean and s.e. over seeds."""
        m = m if m is not None else next(k for k, v in rows.items() if v is rr)
        return reduction(files_of[m], pol, bits, field, add_emb)

    fig, axs = plt.subplots(1, 2, figsize=(COL_W / PRINT_SCALE, 1.5), gridspec_kw=dict(wspace=0.55, width_ratios=[1.5, 1]))
    ax = axs[0]
    learned = [m for m in METHODS if m in rows]
    procs = [('level', 'level heuristic', dict(facecolor='white', edgecolor=REF_EDGE, lw=0.6, hatch='....')),
             ('oracle', 'true-match first', dict(facecolor='white', edgecolor=REF_EDGE, lw=0.6, hatch='////'))]
    ser = [(p, None, st, lab) for p, lab, st in procs] + [('emb', m, bar_style(m, 0.3), NAMES[m].split(' (')[0]) for m in learned]
    w = 0.8 / len(ser)
    for i, (pol, m, st, lab) in enumerate(ser):
        rr = rows[m] if m else rows[learned[-1]]
        mu, se = zip(*[rel(rr, pol, b) for b in bitsL])
        ax.bar(np.arange(len(bitsL)) + offs(len(ser), w)[i], mu, width=w * 0.9, yerr=se,
               error_kw=dict(elinewidth=0.5, ecolor=INK2), zorder=3, label=lab, **st)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(len(bitsL))); ax.set_xticklabels([str(b) for b in bitsL]); ax.set_xlabel('signature width (bits)')
    ax.set_ylabel('fewer refuted calls (%)'); badge_title(ax, 'a', 'Refuted SAT calls')
    ax = axs[1]
    # timing panel: SWEEP_TIME_PREFIX selects the run whose timings are shown (GPU view: the CPU run, where inference
    # ran on one dedicated core; the GPU run's inference shared its GPUs with training jobs)
    _tp = os.environ.get('SWEEP_TIME_PREFIX', 'sweep2_v_')
    tfiles = {m: sorted(glob.glob(f'{_tp}{PREF[m]}_s*.json')) for m in ('base', 'ours')}
    for m in [x for x in ('base', 'ours') if x in learned]:
        a, _ = zip(*[reduction(tfiles[m], 'emb', b, 'time') for b in bitsL])
        b_, _ = zip(*[reduction(tfiles[m], 'emb', b, 'time', add_emb=True) for b in bitsL])
        ax.plot(range(len(bitsL)), a, color=C[m], marker=MARK[m], mfc='white', mew=0.9, ms=2.6, lw=1.0, ls=LS.get(m, '-'))
        ax.plot(range(len(bitsL)), b_, color=C[m], marker=MARK[m], ms=2.6, lw=1.0, ls=LS.get(m, '-'))
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(len(bitsL))); ax.set_xticklabels([str(b) for b in bitsL]); ax.set_xlabel('signature width (bits)')
    ax.set_ylabel('time saved (%)'); badge_title(ax, 'b', 'Sweep time')
    ax.text(0.1, 12, 'SAT time only', fontsize=5.9, color=INK2, va='bottom', ha='left')
    ax.text(0.35, -125, 'including\ninference', fontsize=5.9, color=INK2, va='center', ha='left')
    ax.set_ylim(-150, 30)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.17), fontsize=5.9, handlelength=1.1, columnspacing=0.8)
    pack_row(fig, list(axs), [1.5, 1])
    save(fig, 'fig_sweep')


if __name__ == '__main__':
    os.makedirs('figs', exist_ok=True)
    for w in (sys.argv[1:] or ['recall', 'audit', 'retrieval', 'sweep']):
        globals()[f'fig_{w}'](); print('made', w, flush=True)
