"""Real retrieval data for the Fig. 1 example pair: where the proven equivalent sits in each
model's embedding neighbourhood, and a 2D PCA of that neighbourhood for the figure."""
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys as _sys; _sys.path.insert(0, _os.getcwd())
import pickle, json, numpy as np, torch
from model import AIGEncoder, collate
from evaluate import shash
torch.set_num_threads(1)

D = pickle.load(open('data.pkl', 'rb'))
EX = pickle.load(open('examples.pkl', 'rb'))[0]
_, _, _, n, k, i, j, ca, cb, ptrue, pb_o, pb_v, po_o, po_v = EX
d, var = D['test'][n], D['tvars'][n][k]
s = var['var']


def load(tag, ph):
    m = AIGEncoder(64, 2, proj=ph)
    m.load_state_dict(torch.load(f'v_{tag}.pt'))
    m.eval()
    return m


def emb(m, g):
    with torch.no_grad():
        h, _ = m(collate([g]))
        z = m.project(h) if hasattr(m, 'project') else (m.proj(h) if m.proj is not None else h)
    z = z.numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


def neighbourhood(tag, ph):
    m = load(tag, ph)
    zo, zv = emb(m, d), emb(m, s)
    go = np.arange(1 + d['n_pi'], 1 + d['n_pi'] + len(d['fanins']))   # AND gates of the original
    sims = zo[go] @ zv[j]
    order = np.argsort(-sims)
    rank = int(np.flatnonzero(go[order] == i)[0]) + 1
    # 2D view of the neighbourhood: PCA of the original gates plus the variant gate u
    X = np.vstack([zo[go], zv[j][None]])
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = (X @ Vt[:2].T)
    P = P / (np.abs(P).max() + 1e-9)
    srt = np.sort(sims)[::-1]
    return dict(curve=[float(x) for x in srt],
                rank=rank, n_cand=len(go), sim_true=float(sims[list(go).index(i)]),
                sim_top=float(sims[order[0]]), top_is_true=bool(go[order[0]] == i),
                pts=[[float(a), float(b)] for a, b in P[:-1]], u=[float(P[-1, 0]), float(P[-1, 1])],
                idx_true=int(list(go).index(i)))


out = dict(example=dict(circuit=int(n), family=k, v=int(i), u=int(j), gates_orig=int(ca), gates_var=int(cb),
                        p_true=float(ptrue), base=[float(pb_o), float(pb_v)], ours=[float(po_o), float(po_v)]),
           base=neighbourhood('base_s0', False),
           ours=neighbourhood('eqalign_ph_l0.2_s0', True))
json.dump(out, open('fig1_data.json', 'w'), indent=1)
print(json.dumps({k: (v if k == 'example' else {a: b for a, b in v.items() if a not in ('pts', 'u')})
                  for k, v in out.items()}, indent=1))
