"""Validation metrics (held-out 275 training-circuit cones with their certified variant).
Used ONLY for hyperparameter selection."""
import sys, os, json, pickle, numpy as np, torch
from model import AIGEncoder, collate
from evaluate import shash, gates

torch.set_num_threads(int(os.environ.get('THREADS', 2)))
D = pickle.load(open('data.pkl', 'rb'))
_perm = np.random.default_rng(12345).permutation(len(D['train']))
VAL = np.sort(_perm[:275])


@torch.no_grad()
def run(m, s):
    h, p = m(collate([s])); return m.z(h).numpy(), p.numpy()


def val_metrics(m):
    mae, top, inc, err = [], [], [], []
    for i in VAL:
        d, pr = D['train'][i], D['pairs'][i]
        s = pr['var']
        ho, po = run(m, d); hv, pv = run(m, s)
        go = gates(d)
        mae.append(np.abs(po[go] - d['prob'][go]).mean())
        I, J, P = pr['I'], pr['J'], pr['P']
        if len(I) == 0:
            continue
        rs = shash(d)[I] != shash(s)[J]
        pj = np.where(P == 1, 1 - pv[J], pv[J])
        if rs.any():
            inc.append(np.abs(po[I[rs]] - pj[rs]).mean()); err.append(np.abs(po[I[rs]] - d['prob'][I[rs]]).mean())
        keep = (P == 0) & rs
        if keep.any():
            zo = ho[go] / (np.linalg.norm(ho[go], axis=1, keepdims=True) + 1e-9)
            zj = hv[J[keep]] / (np.linalg.norm(hv[J[keep]], axis=1, keepdims=True) + 1e-9)
            top.append((go[(zj @ zo.T).argmax(1)] == I[keep]).mean())
    return dict(val_mae=float(np.mean(mae)), val_top1_rs=float(np.mean(top)),
                val_incons_rs_ratio=float(np.mean(inc) / np.mean(err)))


if __name__ == '__main__':
    res = json.load(open('val_results.json')) if os.path.exists('val_results.json') else {}
    for tag in sys.argv[1:]:
        from model import build
        m = build(tag)
        m.load_state_dict(torch.load(f'v_{tag}.pt')); m.eval()
        res[tag] = val_metrics(m); print(tag, res[tag], flush=True)
        json.dump(res, open('val_results.json', 'w'), indent=1)
