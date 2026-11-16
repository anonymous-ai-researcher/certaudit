"""E13 large-circuit test suite (test only; no model is trained or tuned on it).

Cones of 1,000 to 6,000 gates from the EPFL designs large enough to contain them: the test designs log2,
sqrt and adder, plus hyp (the confirmatory design; training never used any of the four). Every cone gets
the paper's fixed families S1..S4 and injection at 0.1/0.3/0.5 (seed 30000 + n) and the held-out families
H1/H2 (seed 60000 + n for H2), each SAT-certified, with SAT-proven gate matches, using the paper's own
construction code (core/certdata.py; the held-out job of the paper's heldout.py).

Certifying rewritten arithmetic logic can take arbitrarily long inside the SAT solver, which a Python alarm cannot
interrupt. Each cone therefore runs in its own process and is KILLED after --timeout seconds; such cones, and
cones with any uncertified fixed variant, are dropped and counted in data/large.stats.json (the kept set leans
towards cones whose rewrites SAT can certify, which the paper states). A cone whose held-out construction
crashes keeps its fixed families and has no held-out pair (as two EPFL test circuits in the paper).

  EPFL_DIR=<lsils/benchmarks checkout> python -m tools.gen_large --workers 64      -> data/large.pkl"""
import argparse, os, pickle, time, json
import multiprocessing as mp
import numpy as np

DATA_DIR = os.environ.get('CERT_DATA', 'data')
SRCS = {'arithmetic/log2': 16, 'arithmetic/sqrt': 16, 'arithmetic/adder': 8, 'arithmetic/hyp': 16}
LO, HI, MAX_PI, SEED = 1000, 6000, 1024, 20261104


def job(n, d, out):
    from core.certdata import TKINDS, variant, match, rep_sample, rep_of
    from core.aiglib import to_aig, from_aig, synth, inject, certify
    row = {}
    for k in TKINDS:
        rb, ok = variant(d, k, seed=30_000 + n)
        I, J, P = match(d, rb)
        row[str(k)] = dict(var=rep_sample(rb, d['src']), I=I, J=J, P=P, cert=ok)
    # fixed families first: a native crash in H1's don't-care pass (seen on 2 EPFL test circuits too) then
    # costs only the held-out pair of this cone, exactly as in the paper's EPFL and ISCAS suites
    pickle.dump(row, open(out + '.row.tmp', 'wb')); os.replace(out + '.row.tmp', out + '.row')
    ho = {}
    rep = rep_of(d); a = to_aig(rep)
    for name in ('H1', 'H2'):
        b = synth(a, ['rsdc', 'besop', 'rw6dc', 'rfdc']) if name == 'H1' else inject(rep, 0.3, 60_000 + n, kinds=('xor',))
        ok = certify(a, b); rb = from_aig(b)
        I, J, P = match(d, rb)
        ho[name] = dict(var=rep_sample(rb, d['src']), I=I, J=J, P=P, cert=ok)
    pickle.dump((row, ho), open(out + '.tmp', 'wb')); os.replace(out + '.tmp', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=1200, help='seconds per cone before it is killed and dropped')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    from core.certdata import build_cones
    t0 = time.time(); rng = np.random.default_rng(SEED); cones = []
    for s, k in SRCS.items():
        cones += build_cones([s], k, rng, lo=LO, hi=HI, max_pi=MAX_PI)
    if a.limit:
        cones = cones[:a.limit]
    name = 'large' + ('_smoke' if a.limit else '')
    part = os.path.join(DATA_DIR, name + '_parts'); os.makedirs(part, exist_ok=True)
    print(f'{len(cones)} cones, gates {min(len(c["fanins"]) for c in cones)}..{max(len(c["fanins"]) for c in cones)}', flush=True)
    ctx = mp.get_context('fork')
    done = lambda n: any(os.path.exists(f'{part}/{n:03d}{x}') for x in ('.pkl', '.timeout', '.crashed'))
    todo = [n for n in range(len(cones)) if not done(n)]
    running, timed_out = {}, []
    while todo or running:
        while todo and len(running) < a.workers:
            n = todo.pop(0); p = ctx.Process(target=job, args=(n, cones[n], f'{part}/{n:03d}.pkl')); p.start()
            running[n] = (p, time.time())
        time.sleep(2)
        for n, (p, ts) in list(running.items()):
            if not p.is_alive():
                p.join(); del running[n]
                if not os.path.exists(f'{part}/{n:03d}.pkl'):
                    open(f'{part}/{n:03d}.crashed', 'w').write(str(p.exitcode))
                st = 'done' if os.path.exists(f'{part}/{n:03d}.pkl') else ('fixed families only (held-out crashed)' if os.path.exists(f'{part}/{n:03d}.pkl.row') else 'FAILED')
                print(f'cone {n}: {st} ({time.time() - ts:.0f}s)', flush=True)
            elif time.time() - ts > a.timeout:
                p.kill(); p.join(); del running[n]; timed_out.append(n)
                open(f'{part}/{n:03d}.timeout', 'w').write(str(a.timeout))
                print(f'cone {n}: killed after {a.timeout}s (dropped)', flush=True)
    keep, stats_drop, no_ho = [], dict(timeout=0, failed=0, uncertified=0), 0
    for n in range(len(cones)):
        f = f'{part}/{n:03d}.pkl'
        if os.path.exists(f):
            row, ho = pickle.load(open(f, 'rb'))
        elif os.path.exists(f + '.row'):                 # fixed families done, held-out construction crashed or timed out
            row, ho = pickle.load(open(f + '.row', 'rb')), None
        else:
            stats_drop['timeout' if os.path.exists(f'{part}/{n:03d}.timeout') else 'failed'] += 1; continue
        if not all(v['cert'] for v in row.values()):
            stats_drop['uncertified'] += 1; continue
        if ho is not None and not all(v['cert'] for v in ho.values()):
            ho = None
        no_ho += ho is None
        keep.append((n, row, ho))
    out = dict(test=[cones[n] for n, _, _ in keep], tvars=[r for _, r, _ in keep], ho=[h for _, _, h in keep])
    pickle.dump(out, open(os.path.join(DATA_DIR, name + '.pkl'), 'wb'))
    g = [len(c['fanins']) for c in out['test']] or [0]
    summ = dict(file=name + '.pkl', built=len(cones), kept=len(keep), dropped=stats_drop, kept_without_heldout=no_ho, timeout_s=a.timeout,
                gates_min=min(g), gates_median=int(np.median(g)), gates_max=max(g),
                per_design={s: sum(c['src'] == s for c in out['test']) for s in SRCS},
                built_per_design={s: sum(c['src'] == s for c in cones) for s in SRCS},
                proven_pairs=int(sum(len(v['I']) for r in out['tvars'] for v in r.values())), seconds=round(time.time() - t0))
    json.dump(summ, open(os.path.join(DATA_DIR, name + '.stats.json'), 'w'), indent=1)
    print(json.dumps(summ), flush=True)
