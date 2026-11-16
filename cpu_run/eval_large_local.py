"""Evaluate the paper's trained models (local, 3 seeds) on the E13 large-circuit suite (test only)."""
import sys, os, pickle, torch
import eval_iscas as E
E.S = pickle.load(open('../data/large.pkl', 'rb'))
from model import build
torch.set_num_threads(int(os.environ.get('THREADS', 1)))
for tag in sys.argv[1:]:
    out = f'rawlg_{tag}.pkl'
    if os.path.exists(out):
        continue
    m = build(tag); m.load_state_dict(torch.load(f'v_{tag}.pt')); m.eval()
    pickle.dump(E.evaluate(m), open(out, 'wb')); print('evaluated', tag, flush=True)
