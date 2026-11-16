"""Extra SAT-certified training variants with SAT-proven gate matches, built exactly as the paper built each
cone's training pair (core/certdata.py = the paper's data.py). CPU only, resumable, robust to native crashes.

  python -m tools.gen_extra_variants --family extra --workers 64   # E4: data/pairs_extra.pkl
  python -m tools.gen_extra_variants --family h2    --workers 64   # E9: data/pairs_h2.pkl
  add --limit 20 for the smoke test (writes *_smoke.pkl, which only the smoke jobs use)

extra: up to 3 variants for training cone n, family KINDS[(n + v) % 6] for v = 1, 2, 3 (never the paper's family
       KINDS[n % 6]). Synthesis scripts are deterministic; injection uses seed 100000 * v + n. A variant that is
       structurally identical to the paper's variant or to an earlier extra one is dropped (counted).
h2:    one XOR re-encoding variant per training cone, x&y = (x^y)^(x|y) at rate 0.3 (the held-out family H2 of the
       test set, here applied to TRAINING cones, seed 900000 + n). Test cones are never touched.
Variants that fail certification are dropped and counted. A cone whose processing crashes or exceeds the
timeout twice is recorded with no extra variant."""
import argparse, os, pickle, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import numpy as np

DATA_DIR = os.environ.get('CERT_DATA', 'data')
D = None


def _init():
    global D
    D = pickle.load(open(os.path.join(DATA_DIR, 'data.pkl'), 'rb'))


def _same(r1, r2):
    return r1['n_pi'] == r2['n_pi'] and np.array_equal(r1['fanins'], r2['fanins']) and np.array_equal(r1['pos'], r2['pos'])


def job(n, family):
    import signal
    from core.certdata import KINDS, variant, match, rep_sample, rep_of
    from core.aiglib import to_aig, from_aig, inject, certify

    def _to(*_):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _to); signal.alarm(300)
    d = D['train'][n]; out = []; st = dict(cert_fail=0, duplicate=0)
    try:
        if family == 'extra':
            seen = [D['pairs'][n]['var']]
            for v in (1, 2, 3):
                k = KINDS[(n + v) % len(KINDS)]
                rb, ok = variant(d, k, seed=100_000 * v + n)
                if not ok:
                    st['cert_fail'] += 1; continue
                s = rep_sample(rb, d['src'])
                if any(_same(s, x) for x in seen):
                    st['duplicate'] += 1; continue
                I, J, P = match(d, rb); seen.append(s)
                out.append(dict(var=s, I=I, J=J, P=P, kind=str(k), cert=ok, extra=v))
        else:
            rep = rep_of(d); a = to_aig(rep)
            b = inject(rep, 0.3, 900_000 + n, kinds=('xor',))
            ok = certify(a, b)
            if ok:
                rb = from_aig(b); I, J, P = match(d, rb)
                out.append(dict(var=rep_sample(rb, d['src']), I=I, J=J, P=P, kind="('xor', 0.3)", cert=ok, extra=1))
            else:
                st['cert_fail'] += 1
    finally:
        signal.alarm(0)
    return n, out, st


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--family', choices=['extra', 'h2'], required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count()); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    _init()
    N = a.limit or len(D['train'])
    name = {'extra': 'pairs_extra', 'h2': 'pairs_h2'}[a.family] + ('_smoke' if a.limit else '')
    part = os.path.join(DATA_DIR, name + '.partial.pkl')
    done = pickle.load(open(part, 'rb')) if os.path.exists(part) else {}
    tries = {}; t0 = time.time()
    while True:
        todo = [n for n in range(N) if n not in done]
        if not todo:
            break
        try:
            with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
                futs = {ex.submit(job, n, a.family): n for n in todo}
                for i, f in enumerate(as_completed(futs)):
                    n = futs[f]
                    try:
                        n, out, st = f.result()
                        done[n] = (out, st)
                    except BrokenProcessPool:
                        raise
                    except Exception as e:  # noqa: BLE001  timeout or Python error in this cone
                        tries[n] = tries.get(n, 0) + 1
                        if tries[n] >= 2:
                            done[n] = ([], dict(error=type(e).__name__))
                    if i % 200 == 0:
                        pickle.dump(done, open(part, 'wb'))
                        print(f'{len(done)}/{N} cones, {time.time() - t0:.0f}s', flush=True)
        except BrokenProcessPool:                     # a native crash killed a worker: restart, mark in-flight cones
            for n in todo:
                if n not in done:
                    tries[n] = tries.get(n, 0) + 1
                    if tries[n] >= 3:
                        done[n] = ([], dict(error='crash'))
            print('worker crashed; restarting pool', flush=True)
        pickle.dump(done, open(part, 'wb'))
    X = [done[n][0] for n in range(N)]
    stats = [done[n][1] for n in range(N)]
    pickle.dump(X, open(os.path.join(DATA_DIR, name + '.pkl'), 'wb'))
    nv = sum(len(x) for x in X)
    summ = dict(file=name + '.pkl', cones=N, variants=nv, proven_pairs=int(sum(len(p['I']) for x in X for p in x)),
                cert_fail=sum(s.get('cert_fail', 0) for s in stats), duplicates_dropped=sum(s.get('duplicate', 0) for s in stats),
                cones_failed=sum('error' in s for s in stats), seconds=round(time.time() - t0))
    import json
    json.dump(summ, open(os.path.join(DATA_DIR, name + '.stats.json'), 'w'), indent=1)
    print(json.dumps(summ), flush=True)
