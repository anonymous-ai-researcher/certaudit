"""Fig. 1, version E. Same three-panel story as version D (one function, many certified implementations;
the prediction moves; the representation moves) and the same data, redrawn so that every element is
read from the data and every label is set in the paper's sans-serif figure face.

Changes from D, none of which removes information:
  * (a) The two cones are drawn natively from the real AIGs (fig1_cones.json, positions computed from
    the circuits), and every SAT-proven correspondence between them is shown: an original gate and its
    proven equivalents carry the same number; a dashed ring marks equivalence up to complement (the
    same convention as dashed edges); gray gates of G' have no equivalent in G. D showed only v and u.
  * (b) Unchanged content; numbers and symbols in the sans face, legend drawn with the plot's own marks.
  * (c) The two offset surfaces become two aligned rows on one shared rank axis with ticks, so the
    position of the proven equivalent can be read off; the 63 nearer gates are bracketed and counted.
Design rules: white background, hairline rules; colour encodes model identity only (brick red =
Baseline, cobalt = EqAlign); circuits, proofs and axes are neutral ink.
"""
import json, os, pickle, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open('../fig1_data.json'))
E = D['example']

# proven pairs of this example (CPU-run data, the run the example comes from)
sys.path.insert(0, 'work')
_DATA = pickle.load(open('work/data.pkl', 'rb'))
_EX = pickle.load(open('work/examples.pkl', 'rb'))[0]
n_circ, fam = _EX[3], _EX[4]
assert (n_circ, fam, _EX[5], _EX[6]) == (E['circuit'], E['family'], E['v'], E['u'])
V = _DATA['tvars'][n_circ][fam]
from fig_aig import cone_nodes


def layout(s, root):
    """Level layout of the fanin cone of root (same rule as gen_cones.py): level = longest path from a
    PI inside the cone; gates of one level are spread evenly in index order."""
    base = 1 + s['n_pi']
    nodes = sorted(cone_nodes(s, root))
    lev = {v: 0 for v in nodes if v < base}
    for v in nodes:                       # indices are topological
        if v >= base:
            a, b = s['fanins'][v - base]
            lev[v] = 1 + max(lev[int(a) >> 1], lev[int(b) >> 1])
    maxl = max(lev.values())
    byl = {}
    for v in nodes:
        byl.setdefault(lev[v], []).append(v)
    fin = {v: [int(x) >> 1 for x in s['fanins'][v - base]] for v in nodes if v >= base}
    fout = {}
    for v, fs in fin.items():
        for f in fs:
            fout.setdefault(f, []).append(v)
    order = {Lv: sorted(vs) for Lv, vs in byl.items()}
    def xs(order):
        X = {}
        for Lv, vs in order.items():
            k = len(vs)
            for t, v in enumerate(vs):
                X[v] = (t - (k - 1) / 2) / max(k, 1.6)
        return X
    # barycentre sweeps (up from the PIs, then down from the root) to reduce edge crossings
    for _ in range(4):
        X = xs(order)
        for Lv in range(1, maxl + 1):
            order[Lv] = sorted(order[Lv], key=lambda v: (sum(X[f] for f in fin[v]) / 2, v))
            X = xs(order)
        for Lv in range(maxl - 1, -1, -1):
            order[Lv] = sorted(order[Lv], key=lambda v: (sum(X[g] for g in fout.get(v, [])) / max(len(fout.get(v, [])), 1)
                                                         if fout.get(v) else X[v], v))
            X = xs(order)
    X = xs(order)
    pos = {str(v): (X[v], lev[v] / max(maxl, 1)) for v in nodes}
    edges = []
    for v in nodes:
        if v >= base:
            a, b = s['fanins'][v - base]
            for lit in (a, b):
                edges.append((int(lit) >> 1, v, int(lit) & 1))
    return dict(pos=pos, edges=edges, pis=[v for v in nodes if v < base], root=root, maxl=maxl)


O = layout(_DATA['test'][n_circ], E['v'])
Vc = layout(V['var'], E['u'])          # the cone of u itself, the gate whose prediction (b) and rank (c) are shown
onodes = {int(k) for k in O['pos']}
vnodes = {int(k) for k in Vc['pos']}
PAIRS = [(int(a), int(b), int(p)) for a, b, p in zip(V['I'], V['J'], V['P'])
         if int(a) in onodes and int(b) in vnodes]
assert any(a == E['v'] and b == E['u'] and p == 0 for a, b, p in PAIRS)

W, H = 18.00, 4.95
ZA, ZB, ZC = (0.00, 7.10), (7.36, 11.16), (11.42, 18.00)
DIV = [7.23, 11.29]
TITLE_Y, SUB_Y = 4.88, 4.52

L = []
A = L.append
A(r"""\begin{tikzpicture}[x=1cm,y=1cm, font=\FB, >={Stealth[length=3.4pt]},
  ztitle/.style={font=\FT, text=ink, anchor=north west},
  zsub/.style={font=\FS, text=ink2, anchor=north west},
  ccap/.style={font=\FB, text=ink, anchor=north},
  ccap2/.style={font=\FS, text=ink2, anchor=north},
  card/.style={draw=edge, fill=white, rounded corners=2.5pt, line width=0.55pt},
  ghost/.style={draw=edge, fill=panel, rounded corners=2.5pt, line width=0.55pt},
  badge/.style={rounded corners=1.8pt, inner sep=2.4pt, font=\FK, text=white}]""")
A(r"\useasboundingbox (0,0) rectangle (%.2f,%.2f);" % (W, H))
for x in DIV:
    A(r"\draw[edge, line width=0.5pt] (%.2f,0.08) -- (%.2f,%.2f);" % (x, x, H - 0.30))
IT = lambda s: r'\textit{%s}' % s
for (x0, x1), letter, t, sub in [
        (ZA, 'a', r'One function, many certified implementations',
         r'equal numbers: SAT-proven equivalent gates; gray: none in %s' % IT('G')),
        (ZB, 'b', r'The \emph{prediction} moves', r'same function, two answers'),
        (ZC, 'c', r'The \emph{representation} moves',
         r'gates of %s ranked by cosine similarity to %s; ring: %s' % (IT('G'), IT('u'), IT('v')))]:
    # panel letter in a dark rounded badge, title 10.6 pt after the badge's left edge: the same badge, size and
    # spacing as figstyle.badge_title in the matplotlib figures (7 pt bold white on ink, pad 1.54 pt, radius 1.75 pt)
    A(r"\node[ztitle] (tt%s) at (%.2f,%.2f) {%s};" % (letter, x0 + 0.04 + 10.6 / 28.4528, TITLE_Y, t))
    A(r"\node[fill=ink, text=white, font=\FK, rounded corners=1.75pt, inner sep=1.54pt, anchor=base west] "
      r"at ([xshift=-10.6pt]tt%s.base west) {%s};" % (letter, letter))
    A(r"\node[zsub] at (%.2f,%.2f) {%s};" % (x0 + 0.04, SUB_Y, sub))

# ---------------------------------------------------------------- A: two cones, every proven pair numbered
# node geometry and x positions come from layout_opt.py (fig1_layout.json); y is the logic level
LAY = json.load(open('fig1_layout.json'))
_GP = LAY['orig']['params']
R, R_GRAY, PI_S, LEVEL = _GP['R'], _GP['R_GRAY'], _GP['PI_S'], _GP['LEVEL']
LW_GATE, LW_GRAY = 0.7 / 28.4528, 0.6 / 28.4528      # stroke widths in cm (pt -> cm)
CARD_PAD_X, CARD_PAD_Y = 0.17, 0.16
DECK_N, DECK_DX, DECK_DY = 3, 0.13, 0.10
_pick = os.environ.get('FIG1_PICK')                   # optional: inspect another annealing run


def xs_of(name):
    if _pick and name == 'var':
        return {int(k): v for k, v in LAY['var_all'][int(_pick)]['x'].items()}
    return {int(k): v for k, v in LAY[name]['x'].items()}


XO, XV = xs_of('orig'), xs_of('var')
assert set(XO) == onodes and set(XV) == vnodes, 'layout file does not match the cones'


def card_of(X, h):
    w = max(X.values()) - min(X.values())
    return dict(w=max(w + 2 * R + 2 * CARD_PAD_X, 2.95), h=h + 2 * R + 2 * CARD_PAD_Y)


OH, VH = LEVEL * O['maxl'], LEVEL * Vc['maxl']
C0, C1 = card_of(XO, OH), card_of(XV, VH)
C0['cx'] = 0.22 + C0['w'] / 2
C1['cx'] = C0['cx'] + C0['w'] / 2 + 0.30 + C1['w'] / 2
DECK_TOP = 4.10                         # top of the last ghost card; the zone subtitle ends above it
C1['cy'] = DECK_TOP - DECK_N * DECK_DY - C1['h'] / 2
assert C1['cy'] - C1['h'] / 2 > 0.66, ('cards too low', C1['cy'] - C1['h'] / 2)
C0['cy'] = C1['cy'] - C1['h'] / 2 + C0['h'] / 2
assert C1['cx'] + C1['w'] / 2 + DECK_N * DECK_DX < ZA[1] - 0.10, 'deck overflows zone A'
for k in range(DECK_N, 0, -1):
    A(r"\node[ghost, minimum width=%.3fcm, minimum height=%.3fcm] at (%.3f,%.3f) {};"
      % (C1['w'], C1['h'], C1['cx'] + k * DECK_DX, C1['cy'] + k * DECK_DY))
for C in (C0, C1):
    A(r"\node[card, minimum width=%.3fcm, minimum height=%.3fcm] at (%.3f,%.3f) {};"
      % (C['w'], C['h'], C['cx'], C['cy']))

# numbering: original gates in topological order (by index); a variant gate takes the number of its
# proven equivalent; complemented equivalence is drawn as a dashed ring
og = sorted(v for v in onodes if v not in O['pis'])
num = {v: k + 1 for k, v in enumerate(og)}
vnum = {}
for a, b, p in PAIRS:
    vnum.setdefault(b, (num[a], p))
assert set(num) == {a for a, _, _ in PAIRS}, 'every original gate of the cone has a proven equivalent'


def place(cone, X, C, h, tag):
    mid = (max(X.values()) + min(X.values())) / 2
    y0 = C['cy'] - h / 2
    return {v: (C['cx'] + X[v] - mid, y0 + round(cone['pos'][str(v)][1] * cone['maxl']) * LEVEL) for v in X}


PO = place(O, XO, C0, OH, 'o')
PV = place(Vc, XV, C1, VH, 'w')


PT = 28.4528                               # pt per cm


def dashes(length_cm, on=3.0, off=2.0):
    """'densely dashed' (on 3pt, off 2pt) rescaled so that a stroke of this length starts and ends with a
    dash: no dashed edge leaves its source or meets its arrow head with a gap"""
    Lp = length_cm * PT
    n = max(1, round((Lp + off) / (on + off)))
    k = (Lp + off) / (n * (on + off))
    return 'dash pattern=on %.3fpt off %.3fpt' % (on * k, off * k)


def ring_dashes(r_cm, on=3.0, off=2.0):
    """dash pattern that closes seamlessly around a circle of radius r_cm"""
    Lp = 2 * 3.141592653589793 * r_cm * PT
    n = max(1, round(Lp / (on + off)))
    k = Lp / (n * (on + off))
    return 'dash pattern=on %.3fpt off %.3fpt' % (on * k, off * k)


HEAD_LEN = 2.6 / PT                         # edge arrow head (Stealth, 2.6pt); the stroke stops under it


def outline(kind, ux, uy):
    """distance from a node centre to the outer edge of its drawn outline along the unit vector (ux, uy)"""
    if kind == 'pi':                          # filled square, no stroke
        return (PI_S / 2) / max(abs(ux), abs(uy))
    if kind == 'gray':
        return R_GRAY + LW_GRAY / 2
    return R + LW_GATE / 2


def draw_cone(cone, P, tag, labels):
    """edges are clipped exactly: each starts on the outline of its source and its arrow tip ends on the
    outline of its target (circle or square), so no shaft is hidden under or floats off a node"""
    pis = set(cone['pis'])
    kind = {k: 'pi' if k in pis else ('gate' if labels.get(k) else 'gray') for k in P}
    for a, b, c in cone['edges']:
        (xa, ya), (xb, yb) = P[a], P[b]
        L_ = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
        ux, uy = (xb - xa) / L_, (yb - ya) / L_
        sa, sb = outline(kind[a], ux, uy), outline(kind[b], ux, uy)
        shaft = L_ - sa - sb - 0.675 * HEAD_LEN      # stroke length after the head shortens it (Stealth inset)
        A(r"\draw[ink2, line width=0.5pt%s, -{Stealth[length=2.6pt,width=2.4pt]}] (%.4f,%.4f) -- (%.4f,%.4f);"
          % ((', ' + dashes(shaft)) if c else '', xa + ux * sa, ya + uy * sa, xb - ux * sb, yb - uy * sb))
    for k in pis:
        x, y = P[k]
        A(r"\fill[ink2] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (x - PI_S / 2, y - PI_S / 2, x + PI_S / 2, y + PI_S / 2))
    for k, (x, y) in P.items():
        if k in pis:
            continue
        lab = labels.get(k)
        root = k == cone['root']
        if lab is None:                      # gate of G' with no equivalent in G
            A(r"\filldraw[fill=edge, draw=rule, line width=0.6pt] (%.4f,%.4f) circle (%.4fcm);" % (x, y, R_GRAY))
            continue
        nmb, compl = lab
        fill, txt = ('ink', 'white') if root else ('white', 'ink')
        A(r"\filldraw[fill=%s, draw=ink, line width=0.7pt%s] (%.4f,%.4f) circle (%.4fcm);"
          % (fill, (', ' + ring_dashes(R)) if compl else '', x, y, R))
        A(r"\node[font=\FD, text=%s, inner sep=0] at (%.4f,%.4f) {%d};" % (txt, x, y - 0.005, nmb))
    return kind


draw_cone(O, PO, 'o', {v: (num[v], 0) for v in og})
draw_cone(Vc, PV, 'w', vnum)

cap0 = C0['cy'] - C0['h'] / 2 - 0.08
cap1 = C1['cy'] - C1['h'] / 2 - 0.08
A(r"\node[ccap] at (%.3f,%.3f) {cone of %s in %s, %d gates};" % (C0['cx'], cap0, IT('v'), IT('G'), len(og)))
A(r"\node[ccap2] at (%.3f,%.3f) {%s itself has 122 gates};" % (C0['cx'], cap0 - 0.30, IT('G')))
A(r"\node[ccap] at (%.3f,%.3f) {cone of %s in %s$'$ (I.3), %d gates};"
  % (C1['cx'], cap1, IT('u'), IT('G'), len(vnodes) - len(Vc['pis'])))
A(r"\node[ccap2] at (%.3f,%.3f) {one of \NTotalVar\ certified variants};" % (C1['cx'], cap1 - 0.30))
# proof thread between the two roots
vx, vy = PO[O['root']]
ux, uy = PV[Vc['root']]
A(r"\node[font=\FA, text=ink, anchor=east, inner sep=1.5pt] at (%.3f,%.3f) {%s};" % (vx - R, vy, IT('v')))
A(r"\node[font=\FA, text=ink, anchor=west, inner sep=1.5pt] at (%.3f,%.3f) {%s};" % (ux + R, uy, IT('u')))
# the proof thread rises from the top of v and runs level into the left of u; both tips end on the
# outer edge of the root rings and point along the radius
RT = R + LW_GATE / 2
BEND = 0.35
assert uy - (vy + RT) > BEND + 0.15, 'proof thread: u must sit above v'
_thread = (uy - BEND - (vy + RT)) + 3.141592653589793 / 2 * BEND + (ux - RT - (vx + BEND)) - 2 * 0.675 * 2.8 / PT
A(r"\draw[<->, >={Stealth[length=2.8pt]}, draw=ink, line width=0.65pt, " + dashes(_thread) + "] "
  r"(%.4f,%.4f) -- (%.4f,%.4f) arc[start angle=180, end angle=90, radius=%.3fcm] -- (%.4f,%.4f);"
  % (vx, vy + RT, vx, uy - BEND, BEND, ux - RT, uy))
PROOF_LX = C0['cx'] - C0['w'] / 2           # label sits over the empty corner above card G, left of the deck
A(r"\node[font=\FB, text=ink, anchor=south west, inner sep=2pt] at (%.3f,%.3f) "
  r"{SAT proof: %s\textsubscript{%s} = %s\rlap{$'$}\textsubscript{%s}};"
  % (PROOF_LX, uy + 0.02, IT('f'), IT('v'), IT('f'), IT('u')))

# ---------------------------------------------------------------- B: prediction
BX0, BX1 = ZB[0] + 0.62, ZB[1] - 0.30
bx = lambda t: BX0 + (t - 0.22) / (0.50 - 0.22) * (BX1 - BX0)
AX = 1.02
A(r"\draw[edge, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (bx(0.22), AX, bx(0.50), AX))
for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    A(r"\draw[edge, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (bx(t), AX, bx(t), AX - 0.12))
for t in [0.25, 0.35, 0.45]:
    A(r"\node[font=\FB, text=ink2, anchor=north, inner sep=1.5pt] at (%.3f,%.3f) {%.2f};" % (bx(t), AX - 0.12, t))
A(r"\node[font=\FX, text=ink, anchor=north] at (%.3f,%.3f) {predicted \ooalign{%s\cr\hidewidth\raisebox{-0.25ex}{\kern0.12em\^{}}\hidewidth\cr}};" % ((BX0 + BX1) / 2, AX - 0.55, IT('p')))
TOPLINE = 3.76
A(r"\draw[ink, line width=0.7pt, densely dotted] (%.3f,%.3f) -- (%.3f,%.3f);" % (bx(E['p_true']), AX, bx(E['p_true']), TOPLINE))
A(r"\node[font=\FB, text=ink, anchor=south, inner sep=2pt] at (%.3f,%.3f) {true %s};" % (bx(E['p_true']), TOPLINE, IT('p')))
ROWS = [('Baseline', 'cbase', E['base'], 3.02), ('EqAlign', 'cours', E['ours'], 1.86)]
for nm, c, (a, b), y in ROWS:
    A(r"\node[badge, fill=%s, anchor=north west] at (%.3f,%.3f) {%s};" % (c, BX0, y + 0.78, nm))
    gap = abs(a - b)
    A(r"\draw[%s, line width=2.8pt, line cap=round] (%.3f,%.3f) -- (%.3f,%.3f);" % (c, bx(b), y, bx(a), y))
    if (bx(a) + bx(b)) / 2 + 0.85 < bx(E['p_true']):
        A(r"\node[font=\FA, text=ink, anchor=south, inner sep=4.5pt] at (%.3f,%.3f) {moves %.2f};"
          % ((bx(a) + bx(b)) / 2, y, gap))
    else:
        A(r"\node[font=\FA, text=ink, anchor=east, inner sep=6pt] at (%.3f,%.3f) {moves %.2f};"
          % (bx(min(a, b)), y, gap))
    A(r"\filldraw[fill=white, draw=%s, line width=1.1pt] (%.3f,%.3f) circle (2.8pt);" % (c, bx(b), y))
    A(r"\fill[%s] (%.3f,%.3f) circle (2.2pt);" % (c, bx(a), y))
LY = 1.46
A(r"\fill[ink] (%.3f,%.3f) circle (2.2pt);" % (BX0 + 0.08, LY))
A(r"\node[font=\FB, text=ink, anchor=west, inner sep=0] at (%.3f,%.3f) {on %s};" % (BX0 + 0.22, LY, IT('G')))
A(r"\filldraw[fill=white, draw=ink, line width=1.0pt] (%.3f,%.3f) circle (2.6pt);" % (BX0 + 1.22, LY))
A(r"\node[font=\FB, text=ink, anchor=west, inner sep=0] at (%.3f,%.3f) {on %s$'$};" % (BX0 + 1.40, LY, IT('G')))

# ---------------------------------------------------------------- C: two rows on one rank axis
N = D['base']['n_cand']
X0, X1 = ZC[0] + 1.02, ZC[1] - 0.22
LO, HI = -0.35, 1.0
ROWC = [('base', 'cbase', 'Baseline', 2.30, 3.38), ('ours', 'cours', 'EqAlign', 0.98, 2.06)]
px = lambda k: X0 + (k - 1) / (N - 1) * (X1 - X0)
for key, c, nm, ya, yb in ROWC:
    py = lambda s, ya=ya, yb=yb: ya + (s - LO) / (HI - LO) * (yb - ya)
    for s_, lab in [(0.0, '0'), (0.5, '0.5')]:
        A(r"\draw[edge, line width=0.4pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (X0, py(s_), X1, py(s_)))
        A(r"\node[font=\FB, text=ink2, anchor=east, inner sep=5pt] at (%.3f,%.3f) {%s};" % (X0, py(s_), lab))
    curve = D[key]['curve']
    pts = ' '.join('(%.3f,%.3f)' % (px(k + 1), py(v)) for k, v in enumerate(curve))
    A(r"\fill[%s, opacity=0.10] (%.3f,%.3f) -- plot coordinates {%s} -- (%.3f,%.3f) -- cycle;"
      % (c, px(1), py(LO), pts, px(N), py(LO)))
    A(r"\draw[%s, line width=1.2pt, line join=round] plot coordinates {%s};" % (c, pts))
    A(r"\draw[rule, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (X0, py(LO), X1, py(LO)))
    r, s_ = D[key]['rank'], D[key]['sim_true']
    A(r"\draw[%s, line width=0.6pt, densely dotted] (%.3f,%.3f) -- (%.3f,%.3f);" % (c, px(r), py(LO), px(r), py(s_)))
    A(r"\filldraw[fill=white, draw=%s, line width=1.2pt] (%.3f,%.3f) circle (3.0pt);" % (c, px(r), py(s_)))
    A(r"\fill[%s] (%.3f,%.3f) circle (1.4pt);" % (c, px(r), py(s_)))
    if r > 1:
        yk = py(HI) + 0.10                   # bracket legs stay clear of the curve maximum at rank 1
        A(r"\draw[ink, line width=0.6pt] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f);"
          % (px(1), yk - 0.07, px(1), yk, px(r - 1), yk, px(r - 1), yk - 0.07))
        A(r"\node[font=\FB, text=ink, anchor=south, inner sep=1.5pt] at (%.3f,%.3f) {%d gates nearer to %s than %s};"
          % ((px(1) + px(r - 1)) / 2, yk, r - 1, IT('u'), IT('v')))
        # between the 0 and 0.5 gridlines, right of the dotted drop line: no rule crosses the label
        A(r"\node[font=\FA, text=ink, anchor=west, inner sep=3pt] at (%.3f,%.3f) {rank %d};" % (px(r), py(0.25), r))
        A(r"\node[badge, fill=%s, anchor=south east] at (%.3f,%.3f) {%s};" % (c, X1 - 0.14, py(LO) + 0.06, nm))
    else:
        A(r"\node[font=\FA, text=ink, anchor=west, inner sep=4.5pt] at (%.3f,%.3f) "
          r"{rank 1: %s is the nearest gate};" % (px(r), py(s_), IT('v')))
        A(r"\node[badge, fill=%s, anchor=north east] at (%.3f,%.3f) {%s};" % (c, X1, py(HI), nm))
# shared rank axis
yA = ROWC[1][3] - 0.02
for k in [1, 32, 64, 96, 122]:
    A(r"\draw[ink2, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (px(k), yA, px(k), yA - 0.10))
    A(r"\node[font=\FB, text=ink2, anchor=north, inner sep=1.5pt] at (%.3f,%.3f) {%d};" % (px(k), yA - 0.10, k))
A(r"\node[font=\FX, text=ink, anchor=north] at (%.3f,%.3f) {rank among the 122 gates of %s, most similar first};"
  % ((X0 + X1) / 2, yA - 0.42, IT('G')))
A(r"\node[font=\FX, text=ink, rotate=90, anchor=south] at (%.3f,%.3f) {cosine similarity to \textbf{z}\rlap{$'$}\textsubscript{%s}};"
  % (X0 - 0.52, (ROWC[1][3] + ROWC[0][4]) / 2, IT('u')))
A(r"\end{tikzpicture}")
body = '\n'.join(L) + '\n'
open('optionE.tex', 'w').write(body)
open('work/paper/fig_overview.tex', 'w').write(body)
print('cards', {k: round(v, 2) for k, v in C0.items()}, {k: round(v, 2) for k, v in C1.items()}, 'pairs', PAIRS)
