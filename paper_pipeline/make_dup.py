r"""Share of original EPFL test gates whose 4,096-pattern signature (same polarity) equals that of another gate of
the same circuit: an upper bound on the gates for which Top-1 retrieval could pick a second, functionally equal
original gate (counted as a miss). Appends \DupTwinPct to paper/numbers.tex."""
import os, pickle, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from aiglib import simulate
from data import rep_of, NW
D = pickle.load(open('../data/data.pkl', 'rb'))
dup = tot = 0
for d in D['test']:
    r = rep_of(d); V = simulate(r, NW)
    b = {}
    for i in range(1 + r.n_pi, r.n_nodes):
        b.setdefault(V[i].tobytes(), []).append(i)
    dup += sum(len(x) for x in b.values() if len(x) > 1); tot += r.n_nodes - 1 - r.n_pi
assert len(D['test']) == 376
open('paper/numbers.tex', 'a').write(f'\\newcommand{{\\DupTwinPct}}{{{100 * dup / tot:.1f}}}\n')
print(dup, tot, 100 * dup / tot)
