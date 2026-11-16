"""Option D, final. Depth carries meaning: a staggered deck of certified variants (one function,
many implementations) and two offset similarity surfaces (one plane per model).

Design rules held here:
  * Background is white. Zones are separated by hairline rules, never by filled panels.
  * Colour encodes model identity only: graphite = Baseline (the neutral reference), orange =
    EqAlign (the only warm hue in the paper). Circuits, proofs, axes and cards are neutral ink.
  * Every box position below is computed from the figure's own geometry table, and the rendered
    PDF is then checked for text collisions by check_layout.py.
"""
import json, os, pymupdf
os.chdir(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open('../fig1_data.json'))
E = D['example']
CM = 28.3465

# ---------------------------------------------------------------- geometry table
W, H = 18.00, 4.95
ZA, ZB, ZC = (0.00, 7.10), (7.36, 11.16), (11.42, 18.00)
DIV = [7.23, 11.29]
TITLE_Y, SUB_Y = 4.88, 4.52
CARD_PAD = 0.26          # clearance between the AIG art and its card border
DECK_N, DECK_DX, DECK_DY = 3, 0.15, 0.15
A_LEFT, A_GAP = 0.22, 0.26
PROOF_CLEAR = 0.14       # gap kept between the proof label and the zone subtitle
PROOF_H = 0.21           # measured height of the one-line proof label
SUB_DESC = 0.29          # measured from the rendered PDF: subtitle baseline box depth


def art(path, h):
    """Width in cm of an included AIG at height h, from the file's own bounding box."""
    r = pymupdf.open('../' + path)[0].rect
    return h * r.width / r.height


# Card sizes follow the art; card centres follow from the left margin and the gap, and the two
# cards are then lowered until the proof label clears the subtitle by PROOF_CLEAR.
OH, VH = 1.95, 2.00
MIN_W0, MIN_W1 = 2.95, 3.05   # cards must be at least as wide as their captions
OW, VW = art('figs/st_orig.pdf', OH), art('figs/st_inj.pdf', VH)
C0 = dict(w=max(OW + 2 * CARD_PAD, MIN_W0), h=OH + 2 * CARD_PAD)
C1 = dict(w=max(VW + 2 * CARD_PAD, MIN_W1), h=VH + 2 * CARD_PAD)
C0['cx'] = A_LEFT + C0['w'] / 2
C1['cx'] = C0['cx'] + C0['w'] / 2 + A_GAP + C1['w'] / 2
SUB_BOTTOM = SUB_Y - SUB_DESC                # the subtitle's own descender line
PROOF_Y = SUB_BOTTOM - PROOF_CLEAR - PROOF_H / 2
deck_top_off = DECK_N * DECK_DY
C1['cy'] = PROOF_Y - 0.26 - deck_top_off - C1['h'] / 2
C0['cy'] = C1['cy'] - C1['h'] / 2 + C0['h'] / 2      # bottom-align, so both captions sit on one line
CAP_BOT = min(C0['cy'] - C0['h'] / 2, C1['cy'] - C1['h'] / 2)
assert CAP_BOT - 0.62 > 0.03, 'cards too low for their captions'
assert C1['cx'] + C1['w'] / 2 + DECK_N * DECK_DX < ZA[1] - 0.10, 'deck overflows zone A'

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
for (x0, x1), t, sub in [(ZA, 'a\\, One function, many certified implementations',
                          'a large-gap pair; SAT-proven equivalent'),
                         (ZB, 'b\\, The \\emph{prediction} moves', 'same function, two answers'),
                         (ZC, 'c\\, The \\emph{representation} moves',
                          "rank of the proven equivalent among the 122 gates of $G$")]:
    A(r"\node[ztitle] at (%.2f,%.2f) {%s};" % (x0 + 0.04, TITLE_Y, t))
    A(r"\node[zsub] at (%.2f,%.2f) {%s};" % (x0 + 0.04, SUB_Y, sub))

# ---------------------------------------------------------------- A: the deck
A(r"\node[card, minimum width=%.3fcm, minimum height=%.3fcm] at (%.3f,%.3f) {};"
  % (C0['w'], C0['h'], C0['cx'], C0['cy']))
A(r"\node[inner sep=0] at (%.3f,%.3f) {\includegraphics[height=%.3fcm]{../figs/st_orig.pdf}};"
  % (C0['cx'], C0['cy'], OH))
for k in range(DECK_N, 0, -1):
    A(r"\node[ghost, minimum width=%.3fcm, minimum height=%.3fcm] at (%.3f,%.3f) {};"
      % (C1['w'], C1['h'], C1['cx'] + k * DECK_DX, C1['cy'] + k * DECK_DY))
A(r"\node[card, minimum width=%.3fcm, minimum height=%.3fcm] at (%.3f,%.3f) {};"
  % (C1['w'], C1['h'], C1['cx'], C1['cy']))
A(r"\node[inner sep=0] at (%.3f,%.3f) {\includegraphics[height=%.3fcm]{../figs/st_inj.pdf}};"
  % (C1['cx'], C1['cy'], VH))
cap0 = C0['cy'] - C0['h'] / 2 - 0.10
cap1 = C1['cy'] - C1['h'] / 2 - 0.10
A(r"\node[ccap] at (%.3f,%.3f) {cone of $v$ in $G$, 6 gates};" % (C0['cx'], cap0))
A(r"\node[ccap2] at (%.3f,%.3f) {$G$ itself has 122 gates};" % (C0['cx'], cap0 - 0.30))
A(r"\node[ccap] at (%.3f,%.3f) {cone of $u$ in $G'$ (I.3), 12 gates};" % (C1['cx'], cap1))
A(r"\node[ccap2] at (%.3f,%.3f) {one of \NTotalVar\ certified variants};" % (C1['cx'], cap1 - 0.30))
# proof thread, anchored just above each card
pvx, pvy = C0['cx'] - OW / 2 + 0.30, C0['cy'] + C0['h'] / 2 - 0.24
pux, puy = C1['cx'] - VW / 2 + 0.44, C1['cy'] + C1['h'] / 2 - 0.20
top = PROOF_Y
A(r"\node[font=\FA, text=ink, anchor=east, inner sep=2pt] at (%.3f,%.3f) {$v$};" % (pvx, pvy))
A(r"\node[font=\FA, text=ink, anchor=east, inner sep=2pt] at (%.3f,%.3f) {$u$};" % (pux, puy))
A(r"\draw[<->, >={Stealth[length=2.8pt]}, draw=ink, line width=0.65pt, densely dashed] "
  r"(%.3f,%.3f) .. controls (%.3f,%.3f) and (%.3f,%.3f) .. (%.3f,%.3f);"
  % (pvx + 0.06, pvy + 0.12, pvx + 0.7, top, pux - 0.7, top, pux - 0.02, puy + 0.12))
A(r"\node[fill=white, inner sep=1.8pt, font=\FB, text=ink, anchor=center] at (%.3f,%.3f) "
  r"{SAT proof\,: $f_v=f'_u$};" % ((pvx + pux) / 2, PROOF_Y))

# ---------------------------------------------------------------- B: prediction
BX0, BX1 = ZB[0] + 0.62, ZB[1] - 0.30
bx = lambda t: BX0 + (t - 0.22) / (0.50 - 0.22) * (BX1 - BX0)
AX = 1.02
A(r"\draw[edge, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (bx(0.22), AX, bx(0.50), AX))
for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    A(r"\draw[edge, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (bx(t), AX, bx(t), AX - 0.12))
for t in [0.25, 0.35, 0.45]:
    A(r"\node[font=\FB, text=ink2, anchor=north, inner sep=1.5pt] at (%.3f,%.3f) {%.2f};" % (bx(t), AX - 0.12, t))
A(r"\node[font=\FB, text=ink2, anchor=north] at (%.3f,%.3f) {predicted $\hat p$};" % ((BX0 + BX1) / 2, AX - 0.63))
TOPLINE = 3.76
A(r"\draw[ink, line width=0.7pt, densely dotted] (%.3f,%.3f) -- (%.3f,%.3f);" % (bx(E['p_true']), AX, bx(E['p_true']), TOPLINE))
A(r"\node[font=\FB, text=ink, anchor=south, inner sep=2pt] at (%.3f,%.3f) {true $p$};" % (bx(E['p_true']), TOPLINE))
ROWS = [('Baseline', 'cbase', E['base'], 3.02), ('EqAlign', 'cours', E['ours'], 1.86)]
for nm, c, (a, b), y in ROWS:
    A(r"\node[badge, fill=%s, anchor=north west] at (%.3f,%.3f) {%s};" % (c, BX0, y + 0.78, nm))
    gap = abs(a - b)
    if gap > 0.02:
        A(r"\draw[%s, line width=2.8pt, line cap=round] (%.3f,%.3f) -- (%.3f,%.3f);" % (c, bx(b), y, bx(a), y))
        if (bx(a) + bx(b)) / 2 + 0.85 < bx(E['p_true']):     # centred above the segment, clear of the true-p line
            A(r"\node[font=\FA, text=ink, anchor=south, inner sep=4.5pt] at (%.3f,%.3f) {moves $%.2f$};"
              % ((bx(a) + bx(b)) / 2, y, gap))
        else:                                                  # otherwise left of the segment, as for a small move
            A(r"\node[font=\FA, text=ink, anchor=east, inner sep=6pt] at (%.3f,%.3f) {moves $%.2f$};"
              % (bx(min(a, b)), y, gap))
    else:
        A(r"\node[font=\FA, text=ink, anchor=east, inner sep=6pt] at (%.3f,%.3f) {moves $%.3f$};"
          % (bx(min(a, b)), y, gap))
    A(r"\filldraw[fill=white, draw=%s, line width=1.1pt] (%.3f,%.3f) circle (2.8pt);" % (c, bx(b), y))
    A(r"\fill[%s] (%.3f,%.3f) circle (2.2pt);" % (c, bx(a), y))
A(r"\node[font=\FB, text=ink2, anchor=west, inner sep=0] at (%.3f,%.3f) "
  r"{$\bullet$~on $G$\quad$\circ$~on $G'$};" % (BX0, 1.46))

# ---------------------------------------------------------------- C: two offset surfaces
X0, X1, Y0, Y1 = ZC[0] + 0.98, ZC[1] - 0.52, 0.68, 3.44   # bottom matched to zones A and B
N = D['base']['n_cand']
LO, HI = -0.35, 0.95
DX, DY = 0.38, 0.38
px = lambda k, d: X0 + (k - 1) / (N - 1) * (X1 - X0) + d * DX
py = lambda s, d: Y0 + (s - LO) / (HI - LO) * (Y1 - Y0) + d * DY
for key, c, d in [('base', 'cbase', 1), ('ours', 'cours', 0)]:
    pts = ' '.join('(%.3f,%.3f)' % (px(k + 1, d), py(v, d)) for k, v in enumerate(D[key]['curve']))
    A(r"\fill[%s, opacity=0.09] (%.3f,%.3f) -- plot coordinates {%s} -- (%.3f,%.3f) -- cycle;"
      % (c, px(1, d), py(LO, d), pts, px(N, d), py(LO, d)))
    A(r"\draw[edge, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (px(1, d), py(LO, d), px(N, d), py(LO, d)))
    A(r"\draw[%s, line width=1.3pt, line join=round] plot coordinates {%s};" % (c, pts))
    r, s = D[key]['rank'], D[key]['sim_true']
    A(r"\draw[%s, line width=0.6pt, densely dotted] (%.3f,%.3f) -- (%.3f,%.3f);" % (c, px(r, d), py(LO, d), px(r, d), py(s, d)))
    A(r"\filldraw[fill=white, draw=%s, line width=1.2pt] (%.3f,%.3f) circle (3.0pt);" % (c, px(r, d), py(s, d)))
    A(r"\fill[%s] (%.3f,%.3f) circle (1.4pt);" % (c, px(r, d), py(s, d)))
for s in [-0.25, 0.0, 0.5]:
    A(r"\node[font=\FB, text=ink2, anchor=east, inner sep=3.5pt] at (%.3f,%.3f) {$%s$};" % (px(1, 0), py(s, 0), ('%.2f' % s).rstrip('0').rstrip('.').replace('-', '\\ensuremath{-}') or '0'))
A(r"\node[font=\FB, text=ink2, rotate=90, anchor=south] at (%.3f,%.3f) {$\cos(z'_u,z_\cdot)$};"
  % (X0 - 0.60, (Y0 + Y1) / 2))
A(r"\node[font=\FB, text=ink2, anchor=north] at (%.3f,%.3f) {gates of $G$, most similar first};"
  % ((X0 + X1) / 2 + 0.2, Y0 - 0.26))
A(r"\node[badge, fill=cbase, anchor=south west] at (%.3f,%.3f) {Baseline};" % (px(40, 1), py(0.92, 1)))
A(r"\node[badge, fill=cours, anchor=north west] at (%.3f,%.3f) {EqAlign};" % (px(22, 0), py(0.36, 0)))
A(r"\node[font=\FA, text=ink, anchor=west, align=left, inner sep=5pt] at (%.3f,%.3f) "
  r"{rank 1\\[-1pt]\mdseries\color{ink2}the proven equivalent};" % (px(1, 0), py(D['ours']['sim_true'], 0) - 0.36))
A(r"\node[font=\FA, text=ink, anchor=north, inner sep=6pt] at (%.3f,%.3f) {rank 64};"
  % (px(D['base']['rank'], 1), py(D['base']['sim_true'], 1)))
A(r"\end{tikzpicture}")
body = '\n'.join(L) + '\n'
open('optionD.tex', 'w').write(body)
# the paper includes this from paper/, where the art sits in figs/
open('../paper/fig_overview.tex', 'w').write(body.replace('../figs/', 'figs/'))
print('card0 %.2f x %.2f @ (%.2f,%.2f); card1 %.2f x %.2f @ (%.2f,%.2f); proof y %.2f; deck right %.2f'
      % (C0['w'], C0['h'], C0['cx'], C0['cy'], C1['w'], C1['h'], C1['cx'], C1['cy'], PROOF_Y,
         C1['cx'] + C1['w'] / 2 + DECK_N * DECK_DX))
