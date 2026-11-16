"""E14 exports from a trained model (no training, nothing tuned):
  recall.json   recall@k of the proven equivalent on the EPFL test set, k in 1,2,3,5,10,20,50: for every
                restructured pair with polarity 0 of the 7 fixed families, the rank of the true original gate among
                all original gates by cosine similarity (pessimistic ties), exactly as the paper's Top-1
  fig1.json     the paper's Fig. 1 example pair (data/examples.pkl): rank of the proven equivalent, the sorted
                similarity curve and a 2-D PCA of the neighbourhood
  export.json   marker with counts (the job is done when it exists)

  python -m tools.export_emb --run runs/E1a/gru_d64_r2/eqalign_ph_l0.2_lr0.002_s0 [--fig1]"""
import argparse, json, os, pickle, numpy as np, torch
from core.evaluate import load_run, data, shash, gates, DATA_DIR
from core.model import collate, to_device

KS = [1, 2, 3, 5, 10, 20, 50]


@torch.no_grad()
def emb(m, dev, s):
    h, _ = m(to_device(collate([s]), dev))
    z = m.z(h).float().cpu().numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


def recall(m, dev):
    D = data('data.pkl'); ranks = []
    for n, d in enumerate(D['test']):
        zo = emb(m, dev, d); go = gates(d); sho = shash(d); Zo = zo[go]
        for v in D['tvars'][n].values():
            keep = (v['P'] == 0) & (sho[v['I']] != shash(v['var'])[v['J']])
            if not keep.any():
                continue
            zv = emb(m, dev, v['var'])[v['J'][keep]]
            sims = zv @ Zo.T; true = sims[np.arange(len(zv)), v['I'][keep] - go[0]]
            ranks.append((sims >= true[:, None] - 1e-7).sum(1))
    r = np.concatenate(ranks)
    return dict(k=KS, recall=[float((r <= k).mean() * 100) for k in KS], n_pairs=int(len(r)))


def fig1(m, dev):
    D = data('data.pkl'); EX = pickle.load(open(os.path.join(DATA_DIR, 'examples.pkl'), 'rb'))[0]
    _, _, _, n, k, i, j, ca, cb, ptrue = EX[:10]
    d, s = D['test'][n], D['tvars'][n][k]['var']
    zo, zv = emb(m, dev, d), emb(m, dev, s); go = gates(d)
    sims = zo[go] @ zv[j]; order = np.argsort(-sims)
    X = np.vstack([zo[go], zv[j][None]]); X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False); P = X @ Vt[:2].T; P = P / (np.abs(P).max() + 1e-9)
    with torch.no_grad():
        po = m(to_device(collate([d]), dev))[1].float().cpu().numpy(); pv = m(to_device(collate([s]), dev))[1].float().cpu().numpy()
    return dict(circuit=int(n), family=str(k), v=int(i), u=int(j), p_true=float(ptrue), p_orig=float(po[i]), p_var=float(pv[j]),
                curve=[float(x) for x in np.sort(sims)[::-1]], rank=int(np.flatnonzero(go[order] == i)[0]) + 1,
                n_cand=int(len(go)), sim_true=float(sims[list(go).index(i)]), sim_top=float(sims[order[0]]),
                top_is_true=bool(go[order[0]] == i), pts=[[float(a), float(b)] for a, b in P[:-1]],
                u_pt=[float(P[-1, 0]), float(P[-1, 1])], idx_true=int(list(go).index(i)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--fig1', action='store_true')
    ap.add_argument('--device', default='auto'); a = ap.parse_args()
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    m, cfg = load_run(a.run, dev)
    R = recall(m, dev); json.dump(R, open(os.path.join(a.run, 'recall.json'), 'w'), indent=1)
    out = dict(recall_pairs=R['n_pairs'])
    if a.fig1:
        F = fig1(m, dev); json.dump(F, open(os.path.join(a.run, 'fig1.json'), 'w'), indent=1); out['fig1_rank'] = F['rank']
    json.dump(out, open(os.path.join(a.run, 'export.json'), 'w'), indent=1)
    print('exported', a.run, out, flush=True)
