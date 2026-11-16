"""Stage A of the decision-consequence analysis: per proven pair (v in G, u in G'), the true signal probability
p_v (= p_u up to polarity) and the two predictions, for the 7 fixed families and H1/H2 on the EPFL test set.
Nothing is trained or tuned; models are the CPU run's final models (the GPU export repeats this on the
main-text models).   python3 dump_pair_preds.py base aug ttsim_ph_l0.1_s ...  ->  pairpred/<tag>.npz"""
import os, sys, pickle, numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from model import build, collate
torch.set_num_threads(int(os.environ.get('THREADS', 1)))
D = pickle.load(open('data.pkl', 'rb')); test, tvars = D['test'], D['tvars']
H = pickle.load(open('heldout.pkl', 'rb'))
KINDS = list(tvars[0].keys())
os.makedirs('pairpred', exist_ok=True)


def run(m, s):
    with torch.no_grad():
        _, p = m(collate([s]))
    return p.numpy()


def shash(s):
    from evaluate import shash as _s
    return _s(s)


for tag in sys.argv[1:]:
    out = f'pairpred/{tag}.npz'
    if os.path.exists(out):
        continue
    m = build(tag); m.load_state_dict(torch.load(f'v_{tag}.pt')); m.eval()
    cols = {k: [] for k in ('circ', 'fam', 'p', 'pv', 'pu', 'rs', 'gid')}
    fams = [str(k) for k in KINDS] + ['H1', 'H2']
    for n, d in enumerate(test):
        po = run(m, d); sho = shash(d)
        vs = [(str(k), tvars[n][k]) for k in KINDS]
        if H[n] is not None:
            vs += [(k, H[n][k]) for k in ('H1', 'H2')]
        for k, v in vs:
            I, J, P = v['I'], v['J'], v['P']
            if len(I) == 0:
                continue
            pv = run(m, v['var'])
            pu = np.where(P == 1, 1 - pv[J], pv[J])                       # u's prediction, in v's polarity
            rs = sho[I] != shash(v['var'])[J]                               # restructured pair
            cols['circ'].append(np.full(len(I), n)); cols['fam'].append(np.full(len(I), fams.index(k)))
            cols['p'].append(d['prob'][I]); cols['pv'].append(po[I]); cols['pu'].append(pu)
            cols['rs'].append(rs); cols['gid'].append(I)
    np.savez_compressed(out, fams=np.array(fams), src=np.array([d['src'] for d in test]),
                        **{k: np.concatenate(v) for k, v in cols.items()})
    print('dumped', tag, sum(len(x) for x in cols['p']), 'pairs', flush=True)
