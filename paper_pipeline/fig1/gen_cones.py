"""Layered layouts of the example cone and three of its certified variants, for an isometric figure.
Positions come from the real AIGs; nothing here is drawn by hand."""
import os, sys, json, pickle
os.chdir('work'); sys.path.insert(0, os.getcwd())
import numpy as np
from fig_aig import cone_nodes

D = pickle.load(open('data.pkl', 'rb'))
EX = pickle.load(open('examples.pkl', 'rb'))[0]
n, fam, i, j = EX[3], EX[4], EX[5], EX[6]
d = D['test'][n]


def layout(s, root):
    base = 1 + s['n_pi']
    nodes = sorted(cone_nodes(s, root))
    lev = {}
    for v in nodes:
        if v < base:
            lev[v] = 0
    for v in nodes:
        if v >= base:
            pass
    changed = True
    while changed:
        changed = False
        for v in nodes:
            if v >= base:
                a, b = s['fanins'][v - base]
                pa, pb = int(a) >> 1, int(b) >> 1
                if pa in lev and pb in lev:
                    L = 1 + max(lev[pa], lev[pb])
                    if lev.get(v) != L:
                        lev[v] = L; changed = True
    maxl = max(lev.values())
    byl = {}
    for v in nodes:
        byl.setdefault(lev[v], []).append(v)
    pos = {}
    for L, vs in byl.items():
        k = len(vs)
        for t, v in enumerate(sorted(vs)):
            pos[v] = ((t - (k - 1) / 2) / max(k, 1.6), L / max(maxl, 1))
    edges = []
    for v in nodes:
        if v >= base:
            a, b = s['fanins'][v - base]
            for lit in (a, b):
                p = int(lit) >> 1
                if p in pos:
                    edges.append((p, v, int(lit) & 1))
    return dict(pos={str(k): v for k, v in pos.items()}, edges=edges,
                pis=[v for v in nodes if v < base], root=root,
                n_gates=len([v for v in nodes if v >= base]))


out = {'orig': layout(d, i)}
for key, nm in [("S3", 'S3'), ("('inj', 0.3)", 'inj'), ("H2", 'H2')]:
    V = D['tvars'][n].get(key)
    if V is None:
        continue
    I, J = V['I'], V['J']
    hit = [t for t, x in enumerate(I) if x == i]
    root = int(J[hit[0]]) if hit else int(J[0])
    out[nm] = layout(V['var'], root)
hv = D['heldout'][n] if 'heldout' in D else None
json.dump(out, open('fig1_cones.json', 'w'))
print({k: (v['n_gates'], len(v['pis'])) for k, v in out.items()})
