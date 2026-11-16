"""Emit Option A: one certified rewrite, then the two invariance tests, data-driven."""
import json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open('../fig1_data.json'))
E = D['example']

X0, X1 = 12.30, 17.80          # panel C plot box
Y0, Y1 = 1.00, 3.95
N = D['base']['n_cand']
LO, HI = -0.35, 0.95


def px(k):   # rank 1..N -> x
    return X0 + (k - 1) / (N - 1) * (X1 - X0)


def py(s):   # cosine -> y
    return Y0 + (s - LO) / (HI - LO) * (Y1 - Y0)


def curve(key):
    return ' '.join(f'({px(k+1):.3f},{py(v):.3f})' for k, v in enumerate(D[key]['curve']))


L = []
A = L.append
A(r"""% Option A -- "Two invariance tests". Evidence-first. Colour encodes model identity only
% (gray = Baseline, orange = EqAlign, as everywhere else in the paper); every procedure, axis and
% proof is drawn in neutral ink. Each claim uses the chart type that fits it: a value axis for a
% scalar prediction, a sorted-similarity profile for a ranking.
\begin{tikzpicture}[x=1cm,y=1cm, font=\sffamily\scriptsize, >={Stealth[length=3.2pt]},
  ztitle/.style={font=\sffamily\scriptsize\bfseries, text=ink, anchor=north west},
  zsub/.style={font=\sffamily\tiny, text=ink2, anchor=north west},
  badge/.style={rounded corners=1.5pt, inner sep=2pt, font=\sffamily\tiny\bfseries, text=white}]
\fill[tint, rounded corners=3pt] (0,0) rectangle (5.70,5.30);
\fill[tint, rounded corners=3pt] (5.95,0) rectangle (11.45,5.30);
\fill[tint, rounded corners=3pt] (11.70,0) rectangle (18.00,5.30);
\node[ztitle] at (0.14,5.21) {A\, One certified rewrite};
\node[ztitle] at (6.09,5.21) {B\, The \emph{prediction} moves};
\node[ztitle] at (11.84,5.21) {C\, The \emph{representation} moves};
\node[zsub] at (0.14,4.86) {identity injection $r=0.3$; the pair is SAT-proven};
\node[zsub] at (6.09,4.86) {signal probability of one fixed function};
\node[zsub] at (11.84,4.86) {gates of $G$ sorted by $\cos(z_u,z_\cdot)$};""")

# ---- zone A
A(r"""\node[inner sep=0, anchor=south] (go) at (1.45,0.92) {\includegraphics[height=2.30cm]{../figs/aig_orig.pdf}};
\node[inner sep=0, anchor=south] (gv) at (4.15,0.70) {\includegraphics[height=2.72cm]{../figs/aig_var.pdf}};
\node[text=ink2, font=\sffamily\tiny, anchor=north] at (1.45,0.89) {original $G$, 6 gates};
\node[text=ink2, font=\sffamily\tiny, anchor=north] at (4.15,0.67) {variant $G'$, 14 gates};
\coordinate (rv) at (0.80,3.33); \coordinate (ru) at (3.80,3.58);
\node[font=\sffamily\tiny\bfseries, text=ink, anchor=east, inner sep=2pt] at (rv) {$v$};
\node[font=\sffamily\tiny\bfseries, text=ink, anchor=west, inner sep=2pt] at (ru) {$u$};
\draw[<->, >={Stealth[length=2.8pt]}, draw=ink, line width=0.65pt, densely dashed]
  ($(rv)+(0.10,0.15)$) .. controls (1.70,4.10) and (3.00,4.16) .. ($(ru)+(-0.06,0.16)$);
\node[fill=tint, inner sep=1.5pt, font=\sffamily\tiny, text=ink, anchor=south] at (2.35,4.14)
  {SAT proof\,: $f_v=f'_u$};""")

# ---- zone B
bx = lambda t: 6.95 + (t - 0.20) * 13.6     # 0.20..0.50 -> 6.95..11.03
A(r"\draw[rule, line width=0.5pt] (%.3f,0.78) -- (%.3f,0.78);" % (bx(0.20), bx(0.50)))
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    A(r"\draw[rule, line width=0.5pt] (%.3f,0.78) -- (%.3f,0.66);" % (bx(t), bx(t)))
for t in [0.20, 0.30, 0.40, 0.50]:
    A(r"\node[font=\sffamily\tiny, text=ink2, anchor=north, inner sep=1.5pt] at (%.3f,0.66) {%.2f};" % (bx(t), t))
A(r"\node[font=\sffamily\tiny, text=ink2, anchor=north] at (%.3f,0.30) {predicted $\hat p$};" % bx(0.35))
A(r"\draw[ink, line width=0.7pt, densely dotted] (%.3f,0.78) -- (%.3f,4.26);" % (bx(E['p_true']), bx(E['p_true'])))
A(r"\node[font=\sffamily\tiny, text=ink, anchor=south east, inner sep=2pt] at (%.3f,4.26) {true $p=%.2f$};"
  % (bx(E['p_true']) + 0.06, E['p_true']))
rows = [('Baseline', 'cbase', E['base'], 3.30), ('EqAlign', 'cours', E['ours'], 1.95)]
for nm, c, (a, b), y in rows:
    A(r"\node[badge, fill=%s, anchor=east] at (%.3f,%.2f) {%s};" % (c, bx(0.20) - 0.14, y, nm))
    gap = abs(a - b)
    if gap > 0.02:
        A(r"\draw[%s, line width=2.6pt, line cap=round] (%.3f,%.2f) -- (%.3f,%.2f);" % (c, bx(b), y, bx(a), y))
        A(r"\node[font=\sffamily\tiny\bfseries, text=ink, anchor=south, inner sep=3pt] at (%.3f,%.2f) {moves $%.2f$};"
          % ((bx(a) + bx(b)) / 2, y, gap))
    else:
        A(r"\node[font=\sffamily\tiny\bfseries, text=ink, anchor=east, inner sep=5pt] at (%.3f,%.2f) {moves $%.3f$};"
          % (bx(min(a, b)), y, gap))
    A(r"\filldraw[fill=white, draw=%s, line width=1.1pt] (%.3f,%.2f) circle (2.7pt);" % (c, bx(b), y))
    A(r"\fill[%s] (%.3f,%.2f) circle (2.2pt);" % (c, bx(a), y))
A(r"""\node[font=\sffamily\tiny, text=ink2, anchor=west, inner sep=0] at (6.15,1.22)
  {$\bullet$~$\hat p_v$ on $G$\quad$\circ$~$\hat p'_u$ on $G'$};""")

# ---- zone C: sorted similarity profiles
A(r"\draw[rule, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (X0, Y0, X1, Y0))
A(r"\draw[rule, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (X0, Y0, X0, Y1))
for s in [-0.25, 0.0, 0.25, 0.5, 0.75]:
    A(r"\draw[rule, line width=0.35pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (X0, py(s), X1, py(s)))
    A(r"\node[font=\sffamily\tiny, text=ink2, anchor=east, inner sep=2.5pt] at (%.3f,%.3f) {%s};"
      % (X0, py(s), ('$%.2f$' % s).replace('-', '\\ensuremath{-}')))
for k in [1, 40, 80, 122]:
    A(r"\draw[rule, line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (px(k), Y0, px(k), Y0 - 0.12))
    A(r"\node[font=\sffamily\tiny, text=ink2, anchor=north, inner sep=1.5pt] at (%.3f,%.3f) {%d};" % (px(k), Y0 - 0.12, k))
A(r"\node[font=\sffamily\tiny, text=ink2, anchor=north] at (%.3f,%.3f) {rank among the 122 gates of $G$};"
  % ((X0 + X1) / 2, Y0 - 0.48))
A(r"\node[font=\sffamily\tiny, text=ink2, rotate=90, anchor=south] at (%.3f,%.3f) {$\cos(z_u,z_\cdot)$};"
  % (X0 - 0.52, (Y0 + Y1) / 2))
for key, c, nm in [('base', 'cbase', 'Baseline'), ('ours', 'cours', 'EqAlign')]:
    A(r"\draw[%s, line width=1.2pt, line join=round] plot coordinates {%s};" % (c, curve(key)))
# markers at the proven equivalent
for key, c in [('base', 'cbase'), ('ours', 'cours')]:
    r, s = D[key]['rank'], D[key]['sim_true']
    A(r"\draw[%s, line width=0.6pt, densely dotted] (%.3f,%.3f) -- (%.3f,%.3f);" % (c, px(r), Y0, px(r), py(s)))
    A(r"\filldraw[fill=white, draw=%s, line width=1.2pt] (%.3f,%.3f) circle (2.9pt);" % (c, px(r), py(s)))
    A(r"\fill[%s] (%.3f,%.3f) circle (1.3pt);" % (c, px(r), py(s)))
A(r"\node[font=\sffamily\tiny\bfseries, text=ink, anchor=west, align=left, inner sep=6pt] at (%.3f,%.3f) "
  r"{rank 1\\[-1pt]\mdseries\color{ink2}the proven equivalent};" % (px(1), py(D['ours']['sim_true']) - 0.32))
A(r"\node[font=\sffamily\tiny\bfseries, text=ink, anchor=north, inner sep=5pt] at (%.3f,%.3f) {rank 64};"
  % (px(D['base']['rank']), py(D['base']['sim_true']) - 0.03))
A(r"\node[badge, fill=cbase, anchor=south] at (%.3f,%.3f) {Baseline};" % (px(70), py(0.80)))
A(r"\node[badge, fill=cours, anchor=north west] at (%.3f,%.3f) {EqAlign};" % (px(16), py(0.33)))
A(r"\end{tikzpicture}")
open('optionA.tex', 'w').write('\n'.join(L) + '\n')
print('written', len(L), 'lines')
