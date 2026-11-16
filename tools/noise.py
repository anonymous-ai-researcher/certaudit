"""Seed-to-seed noise floor of the baseline (the paper's noise.py): mean absolute disagreement between
independently trained seeds on the same original gate (restructured matched gates of the 7 fixed families).
  python -m tools.noise --out runs/E1/gru_d64_r2/noise RUN_s0 RUN_s1 ..."""
import argparse, itertools, json, os, numpy as np, torch
from core.evaluate import load_run, shash, data
from core.model import collate, to_device

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('runs', nargs='+')
    ap.add_argument('--device', default='auto'); a = ap.parse_args()
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    miss = [r for r in a.runs if not os.path.exists(os.path.join(r, 'model.pt'))]
    assert not miss, f'noise floor needs every base seed; missing: {miss}'
    ms = [load_run(r, dev)[0] for r in a.runs]
    D = data('data.pkl'); gaps = []
    with torch.no_grad():
        for n, d in enumerate(D['test']):
            sho = shash(d); idx = set()
            for v in D['tvars'][n].values():
                rs = sho[v['I']] != shash(v['var'])[v['J']]
                idx |= set(v['I'][rs].tolist())
            if not idx:
                continue
            idx = np.array(sorted(idx))
            P = [m(to_device(collate([d]), dev))[1].float().cpu().numpy()[idx] for m in ms]
            gaps.append(np.mean([np.abs(P[x] - P[y]).mean() for x, y in itertools.combinations(range(len(ms)), 2)]))
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, 'noise.json.tmp')
    json.dump(dict(noise_floor=float(np.mean(gaps)), n_seeds=len(ms), runs=a.runs), open(tmp, 'w'), indent=1)
    os.replace(tmp, os.path.join(a.out, 'noise.json'))
    print('noise floor', float(np.mean(gaps)), 'from', len(ms), 'seeds', flush=True)
