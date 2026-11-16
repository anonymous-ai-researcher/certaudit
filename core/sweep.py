"""SAT-sweeping probe (the paper's sweep.py run_pair and full_sweep.py driver, unchanged logic).

  python -m core.sweep --run runs/x        # writes runs/x/sweep.json (5 designs x 4 variants, 5 widths, 5 orders)

Every equivalence is decided by a real SAT call (CaDiCaL via PySAT). 'wasted' = refuted SAT calls; it does not
depend on the device. Embedding time is measured on the device used and is reported separately."""
import argparse, json, os, pickle, time, numpy as np, torch
from core.aiglib import simulate, levels
from core.certdata import rep_of
from core.miter import Miter
from core.model import collate, to_device
from core.evaluate import load_run, DATA_DIR

CIRCS = ['random_control/ctrl', 'random_control/router', 'random_control/int2float', 'random_control/cavlc',
         'arithmetic/adder']
KINDS = ['S1', 'S3', ('inj', 0.1), ('inj', 0.3)]


def run_pair(d, s, I, J, P, emb_o=None, emb_v=None, bits=64, policy='random', seed=0, max_cand=None):
    ra, rb = rep_of(d), rep_of(s)
    rng = np.random.default_rng(seed)
    piw = rng.integers(0, 2**63, size=(ra.n_pi, 1), dtype=np.uint64) | \
        (rng.integers(0, 2, size=(ra.n_pi, 1), dtype=np.uint64) << np.uint64(63))
    mask = np.uint64((1 << bits) - 1) if bits < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
    Va = simulate(ra, 1, pi_words=piw)[:, 0] & mask
    Vb = simulate(rb, 1, pi_words=piw)[:, 0] & mask
    la, lb = levels(ra), levels(rb)
    truth = {}
    for i, j, p in zip(I, J, P):
        truth.setdefault(int(j), (int(i), int(p)))
    one = np.uint64(1)
    canon = lambda x: (x ^ mask) if (x & one) else x
    buckets = {}
    for i in range(1 + ra.n_pi, ra.n_nodes):
        buckets.setdefault(int(canon(Va[i])), []).append(i)
    M = Miter(ra, rb)
    va_idx = np.array(M.va) - 1; vb_idx = np.array(M.vb) - 1
    CEa = np.zeros((ra.n_nodes, 0), dtype=bool); CEb = np.zeros((rb.n_nodes, 0), dtype=bool)
    calls = wasted = proved = missing = 0
    t0 = time.perf_counter()
    if emb_o is not None:
        zo = emb_o / (np.linalg.norm(emb_o, axis=1, keepdims=True) + 1e-9)
        zv = emb_v / (np.linalg.norm(emb_v, axis=1, keepdims=True) + 1e-9)
    for j in range(1 + rb.n_pi, rb.n_nodes):
        row = Vb[j]
        cand = buckets.get(int(canon(row)), [])
        if not cand:
            if j in truth:
                missing += 1
            continue
        cand = np.array(cand)
        pol = (Va[cand] != row).astype(np.int64)
        # counterexample refinement (vectorized): drop candidates already distinguished by a CE
        if CEa.shape[1]:
            diff = CEa[cand] != CEb[j][None, :]
            ok = np.where(pol[:, None] == 1, ~diff, diff).any(1) == False
            cand, pol = cand[ok], pol[ok]
        order = np.arange(len(cand))   # 'topo': candidates in topological (index) order, the usual fraig choice
        if policy == 'random':
            rng.shuffle(order)
        elif policy == 'level':
            order = np.lexsort((cand, np.abs(la[cand] - lb[j])))
        elif policy == 'emb':
            order = np.argsort(-(zo[cand] @ zv[j]), kind='stable')
        elif policy == 'oracle':
            tr = truth.get(j)
            order = np.argsort([0 if (tr and c == tr[0]) else 1 for c in cand], kind='stable')
        found = False
        newce = []
        for q in order:
            i, c = int(cand[q]), int(pol[q])
            if newce:  # refine with CEs found while processing this node
                bad = False
                for ca, cb in newce:
                    if (ca[i] != cb) if not c else (ca[i] == cb):
                        bad = True; break
                if bad:
                    continue
            calls += 1
            eq, m = M.check(i, j, c)
            if eq:
                proved += 1; found = True
                break
            wasted += 1
            mv = np.asarray(m) > 0
            ca, cb = mv[va_idx], mv[vb_idx]
            newce.append((ca, cb[j]))
            CEa = np.concatenate([CEa, ca[:, None]], 1); CEb = np.concatenate([CEb, cb[:, None]], 1)
        if j in truth and not found:
            missing += 1
    dt = time.perf_counter() - t0
    return dict(calls=calls, wasted=wasted, proved=proved, missing=missing, time=dt, n_truth=len(truth))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True); p.add_argument('--device', default='auto')
    a = p.parse_args()
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    V = pickle.load(open(os.path.join(DATA_DIR, 'full_variants.pkl'), 'rb'))
    m, cfg = load_run(a.run, dev)

    def emb(s):
        with torch.no_grad():
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t = time.perf_counter(); h, _ = m(to_device(collate([s]), dev))
            z = m.z(h).float().cpu().numpy()
            return z, time.perf_counter() - t
    res = []
    for c in CIRCS:
        eo, to = emb(V[c]['orig'])
        for k in KINDS:
            v = V[c][str(k)]
            ev, tv = emb(v['var'])
            for bits in (4, 8, 16, 32, 64):
                for pol in ('random', 'topo', 'level', 'emb', 'oracle'):
                    for sd in ((0, 1, 2) if pol == 'random' else (0,)):
                        r = run_pair(V[c]['orig'], v['var'], v['I'], v['J'], v['P'], eo, ev, bits=bits, policy=pol, seed=sd)
                        r.update(circ=c, kind=str(k), bits=bits, policy=pol, seed=sd, emb_time=to + tv, model=a.run)
                        res.append(r)
    json.dump(res, open(os.path.join(a.run, 'sweep.json.tmp'), 'w')); os.replace(os.path.join(a.run, 'sweep.json.tmp'), os.path.join(a.run, 'sweep.json'))
    print('sweep done', a.run, flush=True)
