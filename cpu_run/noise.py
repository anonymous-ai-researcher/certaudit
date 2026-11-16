"""Noise floor for self-inconsistency: disagreement between independently trained seeds of the SAME
method on the SAME (original) gate. Compared against delta_rs (same seed, equivalent gate in a rewritten
circuit). Uses the restructured matched original gates of all 7 fixed families."""
import pickle, json, itertools, numpy as np, torch
from model import AIGEncoder, collate
from evaluate import shash

torch.set_num_threads(2)
D = pickle.load(open('data.pkl', 'rb'))
OUT = {}
for name, pref in [('base', 'base'), ('aug', 'aug'), ('ours', 'eqalign_ph_l0.2'), ('ttsim', None)]:
    if pref is None:
        import glob, os
        cands = sorted(glob.glob('v_ttsim_ph_l*_s2.pt'))
        if not cands:
            continue
        pref = cands[0][2:-6]
    ms = []
    for s in range(3):
        m = AIGEncoder(64, 2, proj='_ph' in pref); m.load_state_dict(torch.load(f'v_{pref}_s{s}.pt')); m.eval(); ms.append(m)
    gaps = []
    with torch.no_grad():
        for n, d in enumerate(D['test']):
            sho = shash(d)
            idx = set()
            for v in D['tvars'][n].values():
                rs = sho[v['I']] != shash(v['var'])[v['J']]
                idx |= set(v['I'][rs].tolist())
            if not idx:
                continue
            idx = np.array(sorted(idx))
            P = [m(collate([d]))[1].numpy()[idx] for m in ms]
            gaps.append(np.mean([np.abs(P[a] - P[b]).mean() for a, b in itertools.combinations(range(3), 2)]))
    OUT[name] = float(np.mean(gaps))
    print(name, pref, OUT[name], flush=True)
json.dump(OUT, open('noise_floor.json', 'w'), indent=1)
