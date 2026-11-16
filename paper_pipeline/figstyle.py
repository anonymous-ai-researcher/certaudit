"""Single figure design system for the paper.

Encoding rules (applied in every figure):
  1. Color = identity of the MODEL that produced a number. Same model, same color, in every figure.
       base   brick red     (the model under audit; a real hue, kept apart from the ink of text and axes)
       aug    ochre         (strong baseline)
       tt     teal          (DeepGate2-style functional-similarity baseline)
       circ   teal, outline (FGNN-style circuit-level contrast; same family: representation loss
                             without gate correspondence, so same hue, distinguished by texture)
       l1     lavender      (alignment without negatives)
       ours   cobalt blue   (the method; the only blue in the paper)
     (palette A, validate_palette.js --pairs all: normal-vision dE >= 15.9, CVD dE >= 7.9; ochre is
      2.6:1 on white, so every hue always carries a marker shape and a legend entry or direct label.)
  2. Non-learned procedures (topological order, level heuristic, true-match-first) are drawn as
     OUTLINED ink marks (no fill) or reference lines, never in a model hue: fill = learned model.
  3. Ordered magnitudes (rewrite intensity, lambda) use ONE sequential hue (blue ramp), light to dark.
  4. Text is always ink, never a series color. Identity is never color alone: markers, direct labels,
     or legends accompany every hue.
  5. Hairline recessive grid on the value axis only, no top/right spines, one y-axis per panel."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# One typeface for every figure: URW Nimbus Sans L, the Type 1 face behind LaTeX's helvet (\sffamily), i.e. the face of
# the TikZ overview (Fig. 1). fonts/t1_to_ttf.py converts it to TrueType so matplotlib embeds it (pdf.fonttype 42).
# The face has no Greek rho: math falls back to Computer Modern, the same rho as in the paper's text and captions.
import os as _os
from matplotlib import font_manager as _fm
_FD = _os.path.join(_os.path.dirname(_os.path.realpath(__file__)), 'fonts')
for _w in ('regular', 'bold', 'italic', 'bolditalic'):
    _fm.fontManager.addfont(_os.path.join(_FD, f'nimbussanl-{_w}.ttf'))
FONT = 'Nimbus Sans L TT'

INK, INK2, MUTED, RULE, GRID, SURF = '#1f1e1d', '#4e535a', '#8d939b', '#c7cdd4', '#e7ebef', '#ffffff'
import os as _os
# palette A (v7): cobalt for the method, brick red for the audited baseline, ochre / teal / lavender for the
# other methods. Checked with validate_palette.js --pairs all (5 hues): lightness band, chroma floor, CVD and
# normal-vision separation pass; worst pair brick/teal dE 7.9 deutan, so identity never rests on hue alone.
_G = _os.environ.get('FS_GREEN', '#13a383'); _L1 = _os.environ.get('FS_L1', '#9a6fb5')
C = {'base': '#c0504d', 'aug': '#d4922a', 'tt': _G, 'circ': _G, 'ours': '#2457c5', 'l1': _L1, 'l1tt': _L1}
CIRC_OUTLINE = _os.environ.get('FS_CIRC', 'outline') == 'outline'   # circuit contrast: tint fill + outline instead of hatch


def tint(hexc, f):
    h = hexc.lstrip('#'); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return '#%02x%02x%02x' % tuple(round(c + (255 - c) * f) for c in (r, g, b))


def bar_style(m, lw=0.35):
    """Fill style of a model's bars and legend patches (identity by hue; circuit contrast by texture)."""
    if m == 'circ' and CIRC_OUTLINE:
        return dict(facecolor=tint(C[m], 0.72), edgecolor=C[m], hatch=None, lw=0.6)
    return dict(facecolor=C[m], edgecolor='white', hatch=HATCH.get(m), lw=lw)
# hue = method family: brick no repair | ochre more data | teal representation loss without correspondence |
# lavender alignment without negatives | cobalt contrastive alignment with negatives (ours).
OURS_DARK, OURS_LIGHT = '#1a3f8f', '#eaf0fa'   # cobalt tiers: emphasis text | area tint (marks use C['ours'])
# aqua is 2.74:1 on white: its thin lines are drawn heavier and its outlined markers get a dark aqua rim
EDGE = {}   # teal now passes 3:1, no rim needed
LW_BOOST = {}
# hue = method family; the two representation-loss baselines share aqua and differ by texture/marker/line style
HATCH = {'circ': '////'}
LS = {'circ': (0, (3, 1.5))}
NAMES = {'base': 'Baseline', 'aug': 'Augmentation', 'tt': 'TT-sim', 'circ': 'Circuit contrast', 'ours': 'EqAlign (ours)',
         'l1': 'L1 alignment', 'l1tt': 'L1 alignment + TT-sim'}
MARK = {'base': 'o', 'aug': 's', 'tt': '^', 'circ': 'v', 'ours': 'D', 'l1': 'P', 'l1tt': 'X'}
BLUE = ['#b7d3f6', '#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']   # sequential ramp (unused: blue now marks ours)
NEUTRAL = {'random': '#d6d4ce', 'level': '#52514e', 'oracle': '#1f1e1d'}
REF_EDGE = '#9aa0a8'   # outlined, non-learned reference procedures: recessive grey, never heavier than a model
COL_W, TXT_W = 3.487, 7.140   # IEEEtran conference: columnwidth 252pt, textwidth 516pt

mpl.rcParams.update({
    'font.family': [FONT, 'FreeSans'], 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.edgecolor': RULE, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 0, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'lines.linewidth': 1.4, 'lines.markersize': 3.6, 'legend.frameon': False,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 4,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'custom', 'mathtext.rm': FONT,
    'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold', 'mathtext.fallback': 'cm',
})


def panel_label(ax, s):
    t = ax.get_title(loc='left'); ax.set_title(f'({s})  ' + t, loc='left')


def save(fig, name):
    d = _os.environ.get('FS_DIR', 'figs'); fig.savefig(f'{d}/{name}.pdf'); fig.savefig(f'{d}/{name}.png', dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# v3 additions: structural devices shared by every chart (all neutral; model hues stay identity-only)
BAND = {'syn': '#f5f7f9', 'inj': '#e9edf1', 'ho': '#dde2e8'}


def badge_title(ax, letter, title, y=1.035):
    """Panel letter in a dark rounded badge, followed by the bold panel title (same in every figure)."""
    ax.set_title('')
    t = ax.text(0.0, y, letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=fs('badge'), fontweight='bold',
            color='white', bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.25', fc=INK, ec='none'), zorder=10)
    a = ax.annotate(title, xy=(0.0, y), xycoords='axes fraction', xytext=(9.5, 0), textcoords='offset points',
                    ha='left', va='bottom', fontsize=fs('title'), fontweight='bold', color=INK, annotation_clip=False)
    t._badge = a._badge = True


def family_bands(ax, groups=(('synthesis', -0.5, 3.5, 'syn'), ('injection', 3.5, 6.5, 'inj'), ('held-out', 6.5, 8.5, 'ho')),
                 label_y=None, labels=True, nudge_pt=None):
    """Tinted background bands that group rewrite families, with bracketed group names under the ticks.
    The bracket is placed in POINTS below the x tick labels (tick length + pad + label height + a clear gap),
    so it never touches the tick labels whatever the panel height (label_y is ignored, kept for old calls)."""
    import matplotlib.transforms as mt
    fig = ax.figure
    tick_bottom = mpl.rcParams['xtick.major.size'] + mpl.rcParams['xtick.major.pad'] + fs('tick') * 0.95
    top = tick_bottom + 1.6 / _SC[0]           # printed 1.6 pt of air between tick labels and bracket legs
    bar = top + 1.6 / _SC[0]                   # leg height
    def at(dy_pt):
        return mt.blended_transform_factory(ax.transData, ax.transAxes) + mt.ScaledTranslation(0, -dy_pt / 72, fig.dpi_scale_trans)
    for name, a, b, key in groups:
        ax.axvspan(a, b, facecolor=BAND[key], lw=0, zorder=0)
        if not labels:
            continue
        ax.plot([a + 0.12, b - 0.12], [0, 0], transform=at(bar), color=RULE, lw=0.6, clip_on=False, solid_capstyle='butt')
        for xx in (a + 0.12, b - 0.12):            # legs: from the bar up to `top`, both ends fixed in points
            ax.annotate('', xy=(xx, 0), xycoords=at(top), xytext=(xx, 0), textcoords=at(bar), annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', color=RULE, lw=0.6, shrinkA=0, shrinkB=0))
        tr_lab = at(bar + 1.2 / _SC[0])
        if nudge_pt and name in nudge_pt:          # printed-pt horizontal nudge, for neighbouring names that would touch
            import matplotlib.transforms as mt2
            tr_lab = tr_lab + mt2.ScaledTranslation(nudge_pt[name] / _SC[0] / 72, 0, fig.dpi_scale_trans)
        ax.text((a + b) / 2, 0, name, transform=tr_lab, ha='center', va='top', fontsize=fs('axis'),
                color=INK, clip_on=False)      # the group names are this panel's x-axis name: axis style
    ax.set_xlim(groups[0][1], groups[-1][2])


def end_label(ax, x, y, text, color, dx=4, dy=0, ha='left', fs=None, clear=False):
    """Selective direct label at a line end (text in ink, a short colored tick carries identity)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va='center', fontsize=fs or globals()['fs']('label'),
                color=INK, fontweight='bold' if color == C['ours'] else 'normal', zorder=8,
                bbox=dict(boxstyle='square,pad=0.08', fc='white', ec='none') if clear else None)


def callout(ax, xy, xytext, text, fs=None, ha='center'):
    """Ink callout with a thin leader line; used for the one number a panel exists to show."""
    ax.annotate(text, xy=xy, xytext=xytext, textcoords='data', ha=ha, va='center', fontsize=fs or globals()['fs']('label'), color=INK,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5, shrinkA=1, shrinkB=1.5),
                bbox=dict(boxstyle='round,pad=0.2,rounding_size=0.2', fc='white', ec=RULE, lw=0.4))


def dot(ax, x, mu, sd, m, ms=3.6, lw=0.75):
    """One model's mean (marker) and spread (whisker) at x: filled marker, white rim; circuit contrast hollow."""
    hollow = m == 'circ'
    ax.errorbar(x, mu, yerr=sd, fmt='none', ecolor=C[m], elinewidth=lw, capsize=0, zorder=4, alpha=0.9)
    ax.plot(x, mu, ls='none', marker=MARK[m], ms=ms * (1.1 if m == 'ours' else 1), color=C[m],
            mfc='white' if hollow else C[m], mec=C[m] if hollow else 'white', mew=0.9 if hollow else 0.5, zorder=5 + (m == 'ours'), clip_on=False)


def model_handles(methods, line=False):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=C[m], ls=LS.get(m, '-') if line else 'none', lw=1.1, marker=MARK[m], ms=4,
                   mfc='white' if m == 'circ' else C[m], mec=C[m] if m == 'circ' else 'white',
                   mew=0.9 if m == 'circ' else 0.5, label=NAMES[m]) for m in methods]


# Multi-panel figures are drawn at 1/PRINT_SCALE of their printed width and scaled up by LaTeX, so that
# every figure prints its type at the same size (7 pt rc text -> 7.8 pt on paper, as in earlier versions).
PRINT_SCALE = 1.12

# ---------------------------------------------------------------------------------------------
# Type scale, in PRINTED points (what the reader sees on the page), one table for every figure, the TikZ
# overview included (\FT 8 bold, \FX 7.5, \FB/\FA 7, \FS 6.5, \FK 7 bold in main.tex):
TYPE_PT = dict(title=8.0,   # panel titles (bold)
               axis=7.5,    # axis names
               tick=7.0,    # tick labels and legends
               label=7.0,   # in-plot labels and annotations
               small=6.5,   # secondary notes: family group names, line notes
               badge=7.0)   # panel letter in its badge (bold)
_SC = [PRINT_SCALE]


def fs(role):
    """Font size to pass to matplotlib so that `role` prints at TYPE_PT[role] in the paper."""
    return TYPE_PT[role] / _SC[0]


def set_type(scale=PRINT_SCALE):
    """Declare by how much LaTeX scales this figure (printed width / drawn width) and set rc sizes."""
    _SC[0] = scale
    mpl.rcParams.update({'font.size': fs('label'), 'axes.titlesize': fs('title'), 'axes.labelsize': fs('axis'),
                         'xtick.labelsize': fs('tick'), 'ytick.labelsize': fs('tick'), 'legend.fontsize': fs('tick')})


set_type(PRINT_SCALE)
PANEL_GAP_PT = 9.0 / PRINT_SCALE   # printed gutter between the ink of adjacent panels: exactly 9 pt
TITLE_GAP_PT = 6.0 / PRINT_SCALE   # printed minimum clearance between a title and the next panel's badge


def pack_row(fig, axes, ratios, gap_pt=PANEL_GAP_PT, margin_pt=1.0, verbose=True):
    """Place a row of panels so that the INK gap between neighbours (tick labels, axis labels, band labels;
    titles excluded, they sit above) is exactly gap_pt, and all remaining width goes to the plot areas in
    the given ratios. Titles are then checked; a pair whose titles would come closer than TITLE_GAP_PT gets
    just enough extra gap. Iterates until positions are stable."""
    pt = fig.dpi / 72; W = fig.bbox.width; n = len(axes)
    extra = [0.0] * (n - 1); prev = None
    for _ in range(12):
        fig.canvas.draw(); R = fig.canvas.get_renderer()
        lp, rp = [], []
        for ax in axes:
            badges = [o for o in ax.texts if getattr(o, '_badge', False)]
            for b in badges: b.set_visible(False)
            tb = ax.get_tightbbox(R)
            for b in badges: b.set_visible(True)
            x1 = tb.x1
            if ax is axes[-1] and badges:          # the last title must also stay inside the figure
                x1 = max(x1, max(b.get_window_extent(R).x1 for b in badges))
            lp.append(ax.bbox.x0 - tb.x0); rp.append(x1 - ax.bbox.x1)
        gaps = [gap_pt * pt + e for e in extra]
        avail = W - sum(lp) - sum(rp) - sum(gaps) - 2 * margin_pt * pt
        ws = [avail * r / sum(ratios) for r in ratios]
        x = margin_pt * pt
        for i, ax in enumerate(axes):
            x += lp[i]
            p = ax.get_position(); ax.set_position([x / W, p.y0, ws[i] / W, p.height])
            x += ws[i] + rp[i] + (gaps[i] if i < n - 1 else 0)
        fig.canvas.draw(); R = fig.canvas.get_renderer()
        changed = False
        for i in range(n - 1):
            ti = max(o.get_window_extent(R).x1 for o in axes[i].texts if getattr(o, '_badge', False)) \
                if any(getattr(o, '_badge', False) for o in axes[i].texts) else -1e9
            bj = min(o.get_window_extent(R).x0 for o in axes[i + 1].texts if getattr(o, '_badge', False)) \
                if any(getattr(o, '_badge', False) for o in axes[i + 1].texts) else 1e9
            short = TITLE_GAP_PT * pt - (bj - ti)
            if short > 0.5:
                extra[i] += short; changed = True
        if changed:                                # one gutter width for the whole row
            extra = [max(extra)] * (n - 1)
        pos = [round(v, 2) for v in lp + rp]
        if not changed and pos == prev:
            break
        prev = pos
    if verbose:
        fig.canvas.draw(); R = fig.canvas.get_renderer()
        ink = []
        for i in range(n - 1):
            a0 = axes[i].get_tightbbox(R); a1 = axes[i + 1]
            for b in [o for o in a1.texts if getattr(o, '_badge', False)]: b.set_visible(False)
            for b in [o for o in axes[i].texts if getattr(o, '_badge', False)]: b.set_visible(False)
            g = (a1.get_tightbbox(R).x0 - axes[i].get_tightbbox(R).x1) / pt
            for ax in (axes[i], a1):
                for b in [o for o in ax.texts if getattr(o, '_badge', False)]: b.set_visible(True)
            ink.append(round(g, 2))
        print('panel ink gaps (pt):', ink, '| plot widths (in):', [round(a.get_position().width * fig.get_figwidth(), 3) for a in axes])
