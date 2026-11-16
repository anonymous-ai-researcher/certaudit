"""E16 (decision consequences, stage A): per proven pair (v in G, u in G'), the true probability and both
predictions, for the 7 fixed families and H1/H2 on the EPFL test set. Evaluation only (nothing trained or tuned).
  python -m tools.dump_pair_preds --run runs/E1a/gru_d64_r2/base_lr0.002_s0   ->  <run>/pairpred.npz"""
import argparse, os, numpy as np, torch
from core.evaluate import load_run, data, shash
from core.model import collate, to_device


@torch.no_grad()
def pred(m, dev, s):
    return m(to_device(collate([s]), dev))[1].float().cpu().numpy()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    m, _ = load_run(a.run, dev)
    D = data('data.pkl'); H = data('heldout.pkl'); kinds = list(D['tvars'][0].keys())
    fams = [str(k) for k in kinds] + ['H1', 'H2']
    C = {k: [] for k in ('circ', 'fam', 'p', 'pv', 'pu', 'rs', 'gid')}
    for n, d in enumerate(D['test']):
        po = pred(m, dev, d); sho = shash(d)
        vs = [(str(k), D['tvars'][n][k]) for k in kinds] + ([(k, H[n][k]) for k in ('H1', 'H2')] if H[n] is not None else [])
        for k, v in vs:
            I, J, P = v['I'], v['J'], v['P']
            if len(I) == 0:
                continue
            pv = pred(m, dev, v['var'])
            C['circ'].append(np.full(len(I), n)); C['fam'].append(np.full(len(I), fams.index(k)))
            C['p'].append(d['prob'][I]); C['pv'].append(po[I]); C['pu'].append(np.where(P == 1, 1 - pv[J], pv[J]))
            C['rs'].append(sho[I] != shash(v['var'])[J]); C['gid'].append(I)
    np.savez_compressed(os.path.join(a.run, 'pairpred.npz'), fams=np.array(fams), src=np.array([d['src'] for d in D['test']]),
                        **{k: np.concatenate(v) for k, v in C.items()})
    print('dumped', a.run, sum(len(x) for x in C['p']), 'pairs', flush=True)
