"""Confirmatory test design: EPFL hyp, never used at any point of development.

Built with exactly the main test set's code path (data.build_cones with its default size and
input filters, data._test_row for the seven fixed families with SAT certification and proven
matching, heldout.job for H1/H2). Written to data_hyp.pkl / heldout_hyp.pkl so nothing in the
main results can change. Held-out passes run one cone per subprocess, so a native crash in a
pass loses that cone only."""
import os, sys, pickle, subprocess
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np
import data

N_CONES = 60          # same per-design count as the main test designs
if not os.path.exists('data_hyp.pkl'):
    cones = data.build_cones(['arithmetic/hyp'], N_CONES, np.random.default_rng(20261102))
    rows = [data._test_row((n, d)) for n, d in enumerate(cones)]
    pickle.dump(dict(test=cones, tvars=rows), open('data_hyp.pkl', 'wb'))
    print('fixed families done:', len(cones), 'cones', flush=True)
D = pickle.load(open('data_hyp.pkl', 'rb'))
os.makedirs('ho_hyp', exist_ok=True)
code = ("import sys,pickle; from heldout import job; D=pickle.load(open('data_hyp.pkl','rb')); "
        "n=int(sys.argv[1]); pickle.dump(job((n, D['test'][n])), open(f'ho_hyp/{n:04d}.pkl','wb'))")
fails = []
for n in range(len(D['test'])):
    if not os.path.exists(f'ho_hyp/{n:04d}.pkl'):
        if subprocess.run([sys.executable, '-c', code, str(n)], capture_output=True).returncode != 0:
            fails.append(n)
H = [pickle.load(open(f'ho_hyp/{n:04d}.pkl', 'rb')) if os.path.exists(f'ho_hyp/{n:04d}.pkl') else None
     for n in range(len(D['test']))]
pickle.dump(H, open('heldout_hyp.pkl', 'wb'))
g = [len(d['fanins']) for d in D['test']]
print('hyp cones', len(D['test']), 'gates min/med/max', min(g), int(np.median(g)), max(g),
      '| held-out failures', fails, flush=True)
