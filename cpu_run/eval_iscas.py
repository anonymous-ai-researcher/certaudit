"""Evaluate trained models on the ISCAS suite (fixed + held-out families; no adversarial variants)."""
import sys, os, pickle, numpy as np, torch
import eval2
from model import build
from evaluate import shash, gates
torch.set_num_threads(int(os.environ.get('THREADS', 1)))
S = pickle.load(open('iscas.pkl', 'rb'))

def evaluate(m):
    R = []
    for n, d in enumerate(S['test']):
        ho, po = eval2.run(m, d); go = gates(d); sho = shash(d)
        rec = dict(src=d['src'], mae_orig=float(np.abs(po[go] - d['prob'][go]).mean()), kinds={}, adv=[])
        for k, v in S['tvars'][n].items():
            hv, pv = eval2.run(m, v['var'])
            rec['kinds'][k] = eval2.pair_metrics(d, v['var'], v['I'], v['J'], v['P'], ho, po, hv, pv, sho, n)
        if S['ho'][n] is not None:
            rec['heldout'] = {}
            for k in ('H1', 'H2'):
                v = S['ho'][n][k]; hv, pv = eval2.run(m, v['var'])
                rec['heldout'][k] = eval2.pair_metrics(d, v['var'], v['I'], v['J'], v['P'], ho, po, hv, pv, sho, n)
        R.append(rec)
    return R

if __name__ == '__main__':
    for tag in sys.argv[1:]:
        m = build(tag); m.load_state_dict(torch.load(f'v_{tag}.pt')); m.eval()
        pickle.dump(evaluate(m), open(f'rawis_{tag}.pkl', 'wb')); print('evaluated', tag, flush=True)
