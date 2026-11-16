"""Two-level repair figure: prediction repair (x) versus representation repair (y).

x = restructured inconsistency removed relative to the baseline mean (%); y = restructured Top-1 retrieval (%).
All regimes except the baseline train on the same certified variants. Four of them (augmentation, TT-sim and the
two positives-only L1 alignments) land on the same spot; the main panel draws that spot as one box, and a
magnifier inset resolves the four points. Arrows: baseline -> box, box -> EqAlign.

Every coordinate is computed. Arrow ends are placed GAP points outside the target rim (marker or box edge) and
verified in display space before saving. Usage: python fig_twolevel.py"""
import os, sys, json, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, ConnectionPatch
from figstyle import OURS_DARK, OURS_LIGHT, EDGE, C, INK, INK2, MUTED, RULE, GRID, COL_W, save, fs, set_type
set_type(1.0)   # drawn at the printed column width: matplotlib sizes are printed sizes
from analysis import load, group_summary, SYN, INJ

_e = os.environ.get
P = {'base': 'base', 'aug': 'aug', 'tt': f"ttsim_ph_l{_e('TTLAM', '0.1')}", 'circ': f"circ_ph_l{_e('CIRCLAM', '0.2')}",
     'l1': f"l1align_l{_e('L1LAM', '0.003')}", 'l1tt': f"l1tt_l{_e('L1TTLAM', '0.003')}", 'ours': f"eqalign_ph_l{_e('LAMSTAR', '0.2')}"}
import glob as _glob, re as _re
_seeds = lambda p: sorted(int(_re.search(r'_s(\d+)\.pkl$', f).group(1)) for f in _glob.glob(f'raw_{p}_s*.pkl')
                          if _re.fullmatch(rf'raw_{_re.escape(p)}_s\d+\.pkl', f))
COL = C                                                      # magenta = alignment without negatives
MK = {'base': 'o', 'aug': 's', 'tt': '^', 'circ': 'v', 'l1': 'P', 'l1tt': 'X', 'ours': 'D'}
LAB = {'base': 'Baseline', 'aug': 'Augmentation', 'tt': 'TT-sim', 'circ': 'Circuit contrast',
       'l1': '$L_1$ alignment', 'l1tt': '$L_1$ alignment + TT-sim', 'ours': 'EqAlign (ours)'}
CL = ['aug', 'tt', 'l1', 'l1tt']                            # the cluster resolved by the inset
OURS_INK = OURS_DARK

raw = {k: np.array([[group_summary(load(f'{p}_s{s}'), kinds=SYN + INJ)[0][q] for q in ('incons_rs', 'top1_rs')]
                    for s in _seeds(p)]) for k, p in P.items()}
d0 = raw['base'][:, 0].mean()
pts = {k: np.c_[100 * (d0 - v[:, 0]) / d0, 100 * v[:, 1]] for k, v in raw.items()}
mu = {k: v.mean(0) for k, v in pts.items()}
os.chdir('paper')
json.dump({k: v.tolist() for k, v in mu.items()}, open('figs/fig_twolevel_means.json', 'w'), indent=1)

MS, GAP = 6.0, 1.6
FS, FS_S = fs('label'), fs('small')   # regime names 7 pt; arrow captions and notes 6.5 pt
FS_K = fs('label')                     # regime names in the key: same size as every other legend
fig, ax = plt.subplots(figsize=(COL_W, float(os.environ.get("TL_H", 2.5))))
X0, X1, Y0, Y1 = -6, float(os.environ.get('TL_X1', '36')), 35.5, 67   # x room on the right for the key
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
ax.grid(axis='both', color=GRID, lw=0.5)
ax.set_xlabel('Prediction inconsistency removed vs. baseline (%)', labelpad=2)
ax.set_ylabel('Equivalent-gate retrieval, Top-1 (%)', labelpad=2)
ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([40, 50, 60])


def fit_width(n=3):
    """Resize the plot area so that the saved (tight, padded) figure is exactly COL_W wide: LaTeX then prints it
    at scale 1.0 and every matplotlib size here is the printed size."""
    pad = plt.rcParams['savefig.pad_inches']
    for _ in range(n):
        fig.canvas.draw(); tb = fig.get_tightbbox(fig.canvas.get_renderer())
        W = fig.get_figwidth(); p = ax.get_position()
        li, ri = p.x0 * W - tb.x0, tb.x1 - p.x1 * W
        ax.set_position([(li + 0.05) / W, p.y0, (COL_W - 2 * pad - li - ri) / W, p.height])


fit_width()
ax.axhspan(55, Y1, color=OURS_LIGHT, lw=0, zorder=0)
ax.text(X0 + 0.6, Y1 - 0.8, 'representation partly repaired', ha='left', va='top', fontsize=FS_S, color=OURS_INK,
        style='italic')


def marker(a, k, ms=MS, seeds=True):
    if seeds:
        a.scatter(pts[k][:, 0], pts[k][:, 1], s=5, color=COL[k], alpha=0.35, lw=0, zorder=3)
    hollow = k == 'circ'
    a.plot(*mu[k], marker=MK[k], ms=ms, ls='none', color=COL[k], mfc='white' if hollow else COL[k],
           mec=EDGE.get(k, COL[k]) if hollow else 'white', mew=1.1 if hollow else 0.8, zorder=6)


for k in ('base', 'circ', 'ours'):
    marker(ax, k)

# the cluster box: smallest box around the cluster's seed dots, padded
allc = np.vstack([pts[k] for k in CL])
bx0, bx1 = allc[:, 0].min() - 0.9, allc[:, 0].max() + 0.9
by0, by1 = allc[:, 1].min() - 0.9, allc[:, 1].max() + 0.9
# headroom above the points (box and inset alike, so the magnifier stays exact): room for the inset's labels
by1 = by1 + float(os.environ.get('TL_HEAD', '0')) * (by1 - by0)
box = FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0, boxstyle='round,pad=0,rounding_size=0.6',
                     fc='#eef2f6', ec=INK2, lw=0.7, ls=(0, (2.5, 1.5)), zorder=4)
ax.add_patch(box)
for k in CL:                                                 # small marks only, identity resolved in the inset
    ax.plot(*mu[k], marker=MK[k], ms=3.0, ls='none', color=COL[k], mec='white', mew=0.4, zorder=5)


def disp(a, xy):
    return a.transData.transform(xy)


pt = fig.dpi / 72


def edge_point(src, x_edge=None, y_edge=None):
    """Point on a box edge where the straight line from src first meets it (data coordinates)."""
    if x_edge is not None:
        return np.array([x_edge, src[1] + (by0 + by1) / 2 * 0 + ((by0 + by1) / 2 - src[1]) * 0 + 0])
    return np.array([src[0], y_edge])


# arrow 1: baseline marker -> left edge of the box, aimed at the box's vertical centre
t1 = np.array([bx0, (by0 + by1) / 2])
a1 = FancyArrowPatch(mu['base'], t1, arrowstyle='-|>,head_length=3.2,head_width=2.0', color=INK2, lw=1.1,
                     shrinkA=MS / 2 + GAP, shrinkB=GAP, zorder=5, mutation_scale=1)
ax.add_patch(a1)
# arrow 2: top edge of the box (above EqAlign's x) -> EqAlign marker
s2 = np.array([mu['ours'][0], by1])
a2 = FancyArrowPatch(s2, mu['ours'], arrowstyle='-|>,head_length=3.2,head_width=2.0', color=C['ours'], lw=1.2,
                     shrinkA=GAP, shrinkB=MS / 2 + GAP, zorder=5, mutation_scale=1)
ax.add_patch(a2)

def caption(p, q, text, off, ha, color, va='center'):
    m = (np.asarray(p) + np.asarray(q)) / 2
    ax.annotate(text, xy=m, xytext=off, textcoords='offset points', ha=ha, va=va, fontsize=FS_S, color=color,
                linespacing=1.05)


_cap = os.environ.get('TL_CAP', '2,-11').split(','); caption(mu['base'], t1, '+ certified rewrites', (float(_cap[0]), float(_cap[1])), 'center', INK2, va=_cap[2] if len(_cap) > 2 else 'top')
caption(s2, mu['ours'], '+ contrastive alignment\n(with negatives)', (5, 0), 'left', OURS_INK)
ax.annotate('trained on rewrites,\nno negatives', xy=((bx0 + bx1) / 2, by0), xytext=(0, -2.5),
            textcoords='offset points', ha='center', va='top', fontsize=FS_S, color=INK2, linespacing=1.05)
_bo = [float(x) for x in os.environ.get('TL_BASE', '0,-7.5').split(',')]
ax.annotate('Baseline', xy=mu['base'], xytext=(_bo[0], _bo[1]), textcoords='offset points', ha='center', va='top',
            fontsize=FS, color=INK)
ax.annotate(LAB['circ'], xy=mu['circ'], xytext=(0, 13.5), textcoords='offset points', ha='center', va='bottom',
            fontsize=FS, color=INK, linespacing=1.0)
ax.annotate(LAB['ours'], xy=mu['ours'], xytext=(0, 6.5), textcoords='offset points', ha='center', va='bottom',
            fontsize=FS, color=INK, fontweight='bold')

# magnifier inset in the empty upper-left region, aspect matched to the box
ix0, iy0, iw, ih = 0.035, 0.42, 0.38, 0.45                  # axes fraction
ins = ax.inset_axes([ix0, iy0, iw, ih])
pad_x, pad_y = 0.25, 0.25
LX = 0.50
ins.set_xlim(bx0 + pad_x, bx1 - pad_x); ins.set_ylim(by0 + pad_y, by1 - pad_y)
ins.set_facecolor('#eef2f6'); ins.grid(False)
for s in ins.spines.values():
    s.set_visible(True); s.set_color(INK2); s.set_linewidth(0.7); s.set_linestyle((0, (2.5, 1.5)))
ins.tick_params(labelsize=5.8, length=1.5, pad=1, colors=INK2)
ins.set_xticks([]); ins.set_yticks([])
for k in CL:
    ins.scatter(pts[k][:, 0], pts[k][:, 1], s=7, color=COL[k], alpha=0.4, lw=0, zorder=3)
    ins.plot(*mu[k], marker=MK[k], ms=MS, ls='none', color=COL[k], mec='white', mew=0.8, zorder=6)
# inset labels: left column, slots ordered by y, leader lines to the means
order = sorted(CL, key=lambda k: mu[k][1])
fig.canvas.draw(); fig.canvas.draw(); fig.set_layout_engine('none'); R = fig.canvas.get_renderer()
ibb = ins.get_window_extent(R)
obst = [(ins.transData.transform(p), 3.0 / 2 * pt + 1.5) for k in CL for p in pts[k]] + \
       [(ins.transData.transform(mu[k]), MS / 2 * pt + 2.0) for k in CL]
placed = []
def hits(bb):
    if not (ibb.x0 + 3 < bb.x0 and bb.x1 < ibb.x1 - 3 and ibb.y0 + 3 < bb.y0 and bb.y1 < ibb.y1 - 3):
        return True
    if any(bb.x0 - r < c[0] < bb.x1 + r and bb.y0 - r < c[1] < bb.y1 + r for c, r in obst):
        return True
    return any(bb.x0 < q.x1 + 2 and q.x0 < bb.x1 + 2 and bb.y0 < q.y1 + 2 and q.y0 < bb.y1 + 2 for q in placed)
def seg_hits(a, b):                       # leader segment vs other markers / seed dots / placed labels
    for t in np.linspace(0.08, 0.92, 25):
        c = a + (b - a) * t
        if any(np.hypot(*(c - o)) < r - 1.0 for o, r in obst if np.hypot(*(o - b)) > 1 and np.hypot(*(o - a)) > MS * pt):
            return True
        if any(q.x0 - 1 < c[0] < q.x1 + 1 and q.y0 - 1 < c[1] < q.y1 + 1 for q in placed):
            return True
    return False
def place_column(fx):
    placed.clear(); out = []; tot = 0
    for k in sorted(CL, key=lambda k: -mu[k][1]):       # top to bottom
        m = ins.transData.transform(mu[k]); best = None
        for fy in np.linspace(0.06, 0.94, 45):
            t = ins.text(fx, fy, LAB[k], transform=ins.transAxes, ha='left', va='center', fontsize=FS_K)
            bb = t.get_window_extent(R); t.remove()
            anchor = np.array([bb.x0 - 1.2 * pt, (bb.y0 + bb.y1) / 2])
            if hits(bb) or seg_hits(m, anchor):
                continue
            L = np.hypot(*(anchor - m)); cost = L + 0.3 * abs(anchor[1] - m[1]) + (4 * (10 * pt - L) if L < 10 * pt else 0)
            if best is None or cost < best[0]:
                best = (cost, fy, bb)
        if best is None:
            return None
        placed.append(best[2]); out.append((k, best[1])); tot += best[0]
    return tot, out
_cands = [r + (fx,) for fx in np.linspace(0.40, 0.66, 14) if (r := place_column(fx))]
if _cands and not os.environ.get('TL_LEGEND'):
    _, _sl, LXc = min(_cands, key=lambda z: z[0])
    for k, fy in _sl:
        ins.annotate(LAB[k], xy=mu[k], xytext=(LXc, fy), textcoords='axes fraction', ha='left', va='center',
                     fontsize=FS_K, color=INK, arrowprops=dict(arrowstyle='-', color=COL[k], lw=0.6, shrinkA=1.2,
                                                               shrinkB=MS / 2 + 0.8), zorder=7)
else:
    # no clean label column (dense seeds): a key names each marker instead of leader lines
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=MK[k], ms=MS - 1, ls='none', color=COL[k], mec='white', mew=0.6, label=LAB[k].replace(' + ', '\n+ '))
          for k in sorted(CL, key=lambda k: -mu[k][1])]
    # the key sits in the empty space right of the dashed box, whose four regimes it names
    ax.legend(handles=hs, loc='center left', bbox_to_anchor=(bx1 + 0.4, (by0 + by1) / 2), bbox_transform=ax.transData,
              fontsize=FS_K, frameon=True, facecolor='white', edgecolor='none', framealpha=1.0, borderpad=0.25, handletextpad=0.2, borderaxespad=0, labelspacing=0.3)
# connector lines from the box's two left corners to the inset's two lower/upper right corners
for (xb, yb), (xi, yi) in [((bx0, by1), (1, 1)), ((bx0, by0), (1, 0))]:
    cp = ConnectionPatch(xyA=(xb, yb), coordsA=ax.transData, xyB=(xi, yi), coordsB=ins.transAxes,
                         color=MUTED, lw=0.5, ls=(0, (1.5, 1.5)), zorder=2)
    fig.add_artist(cp)

fit_width(6)                       # the key right of the box may reach past the axes: fit the whole figure again
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_lg = ax.get_legend()
if _lg is not None:                # the key must sit inside the plot area and clear every other text
    _lb = _lg.get_window_extent(_r); _ab = ax.get_window_extent(_r)
    if _lb.x1 > _ab.x1 - 1 or _lb.y0 < _ab.y0 + 1:
        print('KEY OUTSIDE AXES'); _keybad = 1
    else:
        _keybad = 0
    for t in fig.findobj(match=lambda o: o.__class__.__name__ in ('Text', 'Annotation')):
        if t.get_text().strip() and t.get_visible() and t not in _lg.get_texts() and plt.Text.get_window_extent(t, _r).overlaps(_lb):
            print('TEXT UNDER KEY:', repr(t.get_text())); _keybad += 1
else:
    _keybad = 0
# captions must keep printed air between them (>= 2 pt)
_caps = [t for t in ax.texts if t.get_text().strip()]
for i in range(len(_caps)):
    for j in range(i + 1, len(_caps)):
        a_, b_ = plt.Text.get_window_extent(_caps[i], _r), plt.Text.get_window_extent(_caps[j], _r)
        if a_.x0 - 2 * pt < b_.x1 and b_.x0 - 2 * pt < a_.x1 and a_.y0 - 1 * pt < b_.y1 and b_.y0 - 1 * pt < a_.y1:
            print('TEXTS TOO CLOSE:', repr(_caps[i].get_text()), repr(_caps[j].get_text())); _keybad += 1
# exactness report in display points
for name, (p, q, ga, gb) in {'base->box': (mu['base'], t1, MS / 2 + GAP, GAP),
                             'box->EqAlign': (s2, mu['ours'], GAP, MS / 2 + GAP)}.items():
    L = np.linalg.norm(disp(ax, q) - disp(ax, p)) / pt
    print(f'{name}: anchor distance {L:.1f} pt; tail {ga:.1f} pt off source, tip {gb:.1f} pt off target rim')
# every text box must clear every mean marker (display space)
r = fig.canvas.get_renderer(); bad = 0
mk = [(ax, mu[k], MS) for k in ('base', 'circ', 'ours')] + [(ins, mu[k], MS) for k in CL]
for t in fig.findobj(match=lambda o: hasattr(o, 'get_window_extent') and o.__class__.__name__ in ('Text', 'Annotation')):
    if not t.get_text().strip() or not t.get_visible():
        continue
    bb = plt.Text.get_window_extent(t, r)
    for a, xy, ms in mk:
        c = a.transData.transform(xy); rr = ms / 2 * pt
        if bb.x0 - rr < c[0] < bb.x1 + rr and bb.y0 - rr < c[1] < bb.y1 + rr:
            bad += 1; print('TEXT OVER MARKER:', repr(t.get_text()), xy)
seeds = [(ax, p, 2.6) for k in P if k not in CL for p in pts[k]] + [(ins, p, 3.0) for k in CL for p in pts[k]]
ib = ins.get_window_extent(r)
for t in fig.findobj(match=lambda o: o.__class__.__name__ in ('Text', 'Annotation')):
    if not t.get_text().strip() or not t.get_visible() or t.axes is None:
        continue
    bb = plt.Text.get_window_extent(t, r)
    for a, xy, d in seeds:
        if a is not t.axes:
            continue
        c = a.transData.transform(xy); rr = d / 2 * pt
        if a.get_window_extent(r).contains(*c) and bb.x0 - rr < c[0] < bb.x1 + rr and bb.y0 - rr < c[1] < bb.y1 + rr:
            bad += 1; print('TEXT OVER SEED DOT:', repr(t.get_text()), np.round(xy, 2))
    if t.axes is ax and bb.overlaps(ib):
        bad += 1; print('TEXT UNDER INSET:', repr(t.get_text()))
    if t.axes is ax and bb.overlaps(box.get_window_extent(r)) and 'trained on rewrites' not in t.get_text():
        bad += 1; print('TEXT UNDER BOX:', repr(t.get_text()))
    if t.axes is ins and not (ib.x0 + 2 < bb.x0 and bb.x1 < ib.x1 - 2 and ib.y0 + 2 < bb.y0 and bb.y1 < ib.y1 - 2):
        bad += 1; print('INSET TEXT OUTSIDE FRAME:', repr(t.get_text()))
print('text-marker/seed/inset problems:', bad + _keybad)
_tb = fig.get_tightbbox(fig.canvas.get_renderer())
print('saved width %.2f pt (column %.2f pt)' % ((_tb.width + 2 * plt.rcParams['savefig.pad_inches']) * 72, COL_W * 72))
save(fig, 'fig_twolevel')
print({k: np.round(v, 2).tolist() for k, v in mu.items()}, 'box', [round(bx0, 2), round(bx1, 2), round(by0, 2), round(by1, 2)])
