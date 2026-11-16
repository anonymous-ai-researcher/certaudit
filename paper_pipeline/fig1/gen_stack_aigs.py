"""Render the original cone and three of its certified variants as separate vector AIGs,
for the stacked-card figure. Roots are the SAT-proven counterparts of the same gate v."""
import os, sys, pickle
os.chdir('work'); sys.path.insert(0, os.getcwd())
from fig_aig import to_dot, cone_nodes

D = pickle.load(open('data.pkl', 'rb'))
EX = pickle.load(open('examples.pkl', 'rb'))[0]
n, i = EX[3], EX[5]
d = D['test'][n]
print('orig gates', to_dot(d, i, 'orig', 'figs/st_orig'))
for key, nm in [('S3', 'st_s3'), ("('inj', 0.3)", 'st_inj'), ('H2', 'st_h2')]:
    V = D['tvars'][n].get(key)
    if V is None:
        print('skip', key); continue
    I, J = V['I'], V['J']
    hit = [t for t, x in enumerate(I) if x == i]
    root = int(J[hit[0]]) if hit else int(J[0])
    print(nm, to_dot(V['var'], root, nm, f'figs/{nm}'))
