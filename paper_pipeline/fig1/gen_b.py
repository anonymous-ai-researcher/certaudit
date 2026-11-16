"""Option B: the audit spine. Pipeline-first, but every stage is drawn as the thing it is
(a rewrite taxonomy as a banded bar, certification as a miter, matching as a proof per pair)
rather than as a text box. Colour still encodes model identity only; the spine is neutral ink."""
import json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open('../numbers.json'))
L = []
A = L.append
A(r"""\begin{tikzpicture}[x=1cm,y=1cm, font=\sffamily\scriptsize, >={Stealth[length=3.2pt]},
  st/.style={font=\sffamily\tiny\bfseries, text=ink, anchor=north west},
  sub/.style={font=\sffamily\tiny, text=ink2, anchor=north west},
  arr/.style={->, draw=ink2, line width=0.7pt},
  badge/.style={rounded corners=1.5pt, inner sep=2pt, font=\sffamily\tiny\bfseries, text=white}]
\fill[tint, rounded corners=3pt] (0,1.52) rectangle (18.00,5.35);
\fill[tint, rounded corners=3pt] (0,0) rectangle (18.00,1.32);

% ---------- stage 1: the circuit ----------
\node[st] at (0.14,5.26) {1\, Circuit};
\node[inner sep=0, anchor=north] at (1.35,4.84) {\includegraphics[height=1.62cm]{../figs/aig_orig.pdf}};
\node[sub] at (0.14,3.05) {a fanin cone $G$ of an};
\node[sub] at (0.14,2.78) {EPFL design, gate $v$};
\draw[arr] (2.55,3.60) -- (3.15,3.60);

% ---------- stage 2: rewrite taxonomy as a banded bar ----------
\node[st] at (3.30,5.26) {2\, Rewrite into certified variants};
\fill[tint2, rounded corners=1.5pt] (3.30,3.95) rectangle (5.55,4.62);
\fill[tint2, rounded corners=1.5pt] (5.70,3.95) rectangle (7.05,4.62);
\draw[rule, line width=0.5pt, rounded corners=1.5pt, pattern=north east lines, pattern color=rule]
      (7.20,3.95) rectangle (8.25,4.62);
\foreach \x/\t in {3.58/S1, 4.14/S2, 4.70/S3, 5.26/S4}{%
  \node[font=\sffamily\tiny, text=ink, anchor=center] at (\x,4.28) {\t};}
\foreach \x/\t in {5.93/{I.1}, 6.37/{I.2}, 6.81/{I.3}}{%
  \node[font=\sffamily\tiny, text=ink, anchor=center] at (\x,4.28) {\t};}
\foreach \x/\t in {7.46/H1, 7.99/H2}{%
  \node[font=\sffamily\tiny, text=ink, anchor=center] at (\x,4.28) {\t};}
\node[font=\sffamily\tiny, text=ink2, anchor=north] at (4.42,3.90) {synthesis};
\node[font=\sffamily\tiny, text=ink2, anchor=north] at (6.37,3.90) {redundancy};
\node[font=\sffamily\tiny, text=ink2, anchor=north] at (7.72,3.90) {held-out};
\node[sub] at (3.30,3.42) {@NVAR@ certified variants of the test set};
\node[sub] at (3.30,3.08) {$+$ 16 adversarial compositions per circuit};
\draw[arr] (8.50,4.28) -- (9.10,4.28);

% ---------- stage 3: certification as a miter ----------
\node[st] at (9.25,5.26) {3\, Certify, then prove each pair};
\node[draw=ink2, line width=0.5pt, rounded corners=1.5pt, fill=white, minimum width=0.62cm,
      minimum height=0.34cm, font=\sffamily\tiny, inner sep=1pt] (mg) at (9.75,4.80) {$G$};
\node[draw=ink2, line width=0.5pt, rounded corners=1.5pt, fill=white, minimum width=0.62cm,
      minimum height=0.34cm, font=\sffamily\tiny, inner sep=1pt] (mv) at (9.75,4.16) {$G'$};
\node[draw=ink, line width=0.6pt, circle, inner sep=0.6pt, font=\sffamily\tiny, fill=white] (mx) at (10.85,4.48) {$\oplus$};
\draw[arr] (mg) -- (mx); \draw[arr] (mv) -- (mx);
\draw[arr] (mx) -- (11.75,4.48);
\node[font=\sffamily\tiny, text=ink, anchor=west, align=left] at (11.80,4.48) {UNSAT\\[-1pt]\color{ink2}outputs agree};
\node[sub] at (9.25,3.82) {then one SAT call per gate pair:};
\node[sub] at (9.25,3.50) {\textbf{@NPAIR@ proven pairs}; simulation};
\node[sub] at (9.25,3.22) {signatures alone mislabel 1.9\%};
\draw[arr] (13.55,4.28) -- (14.15,4.28);""".replace("@NVAR@", M["NTotalVar"]).replace("@NPAIR@", M["NTestPairsK"]))

# ---------- stage 4: the two measurement heads
A(r"""\node[st] at (14.30,5.26) {4\, Two measurements};
\node[draw=rule, fill=white, rounded corners=2pt, line width=0.5pt, minimum width=3.5cm,
      minimum height=0.74cm, align=left, font=\sffamily\tiny, inner sep=3pt, anchor=north west] at (14.30,4.90)
  {\textbf{Prediction level}\\ $\delta(v,u)$ and $\rho=\bar\delta/\bar\epsilon$};
\node[draw=rule, fill=white, rounded corners=2pt, line width=0.5pt, minimum width=3.5cm,
      minimum height=0.74cm, align=left, font=\sffamily\tiny, inner sep=3pt, anchor=north west] at (14.30,3.92)
  {\textbf{Representation level}\\ Top-1 retrieval, AUROC$_\mathrm{h}$};""")

# ---------- repair loop
A(r"""\draw[cours, line width=0.8pt, ->, >={Stealth[length=3pt]}, rounded corners=3pt]
  (11.30,3.05) -- (11.30,2.42) -- (16.05,2.42) -- (16.05,3.16);
\node[badge, fill=cours, anchor=center] at (13.10,2.42) {EqAlign: InfoNCE on the proven pairs};""")

# ---------- results ribbon
def ribbon(x, title, a, b, fmt, better_down):
    A(r"\node[font=\sffamily\tiny\bfseries, text=ink, anchor=west] at (%.2f,0.92) {%s};" % (x, title))
    A(r"\node[badge, fill=cbase, anchor=west] at (%.2f,0.46) {Baseline %s};" % (x, fmt % a))
    A(r"\draw[arr] (%.2f,0.46) -- (%.2f,0.46);" % (x + 2.05, x + 2.45))
    A(r"\node[badge, fill=cours, anchor=west] at (%.2f,0.46) {EqAlign %s};" % (x + 2.52, fmt % b))

ribbon(0.30, 'Consistency under synthesis $\\rho\\downarrow$', M['BaseRhoSyn'], M['OursRhoSyn'], '%s', True)
ribbon(6.35, 'Retrieval of the proven equivalent $\\uparrow$', M['BaseTopRS'] + '\\%', M['OursTopRS'] + '\\%', '%s', False)
ribbon(12.40, 'Held-out re-encoding $\\rho\\downarrow$', M['BaseRhoHo'], M['OursRhoHo'], '%s', True)
A(r"\node[font=\sffamily\tiny, text=ink2, anchor=west] at (16.55,0.92) {still broken};")
A(r"\end{tikzpicture}")
open('optionB.tex', 'w').write('\n'.join(L) + '\n')
print('ok')
