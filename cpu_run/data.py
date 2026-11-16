"""Dataset construction: cone extraction from EPFL circuits, labels by random simulation,
certified-equivalent variants, and cross-variant node matching by simulation signatures."""
import numpy as np, pickle, sys, time
from aigverse.io import read_aiger_into_aig
from aiglib import Rep, from_aig, to_aig, simulate, popcount_prob, levels, synth, inject, certify

B = '${WORK}/bench/epfl'
TRAIN_SRC = ['arithmetic/bar', 'arithmetic/max', 'arithmetic/sin', 'arithmetic/multiplier', 'arithmetic/square',
             'arithmetic/div', 'random_control/arbiter', 'random_control/mem_ctrl', 'random_control/voter',
             'random_control/i2c', 'random_control/priority', 'random_control/dec']
TEST_SRC = ['arithmetic/log2', 'arithmetic/sqrt', 'arithmetic/adder', 'random_control/cavlc',
            'random_control/ctrl', 'random_control/int2float', 'random_control/router']
NW = 64  # 4096 random patterns


def cone(rep, root, max_gates, rng):
    """Transitive fanin cone of gate `root` (node id), truncated at max_gates (BFS, random order)."""
    base = 1 + rep.n_pi
    inside = set()
    frontier = [root]
    while frontier and len(inside) < max_gates:
        k = int(rng.integers(len(frontier)))
        v = frontier.pop(k)
        if v in inside or v < base:
            continue
        inside.add(v)
        a, b = rep.fanins[v - base]
        frontier += [int(a) >> 1, int(b) >> 1]
    gates = sorted(inside)
    # leaves: fanins outside the cone (excluding const)
    leaves = []
    for v in gates:
        for l in rep.fanins[v - base]:
            u = int(l) >> 1
            if u != 0 and u not in inside and u not in leaves:
                leaves.append(u)
    leaves.sort()
    m = {0: 0}
    for k, u in enumerate(leaves):
        m[u] = 1 + k
    fan = []
    for k, v in enumerate(gates):
        m[v] = 1 + len(leaves) + k
        a, b = rep.fanins[v - base]
        fan.append([2 * m[int(a) >> 1] + (int(a) & 1), 2 * m[int(b) >> 1] + (int(b) & 1)])
    # POs: gates in the cone with no fanout inside the cone (at least root)
    used = set()
    for a, b in fan:
        used.add(int(a) >> 1); used.add(int(b) >> 1)
    pos = [2 * m[v] for v in gates if m[v] not in used]
    return Rep(len(leaves), fan, pos)


def sample(rep, label_seed=0):
    V = simulate(rep, NW, seed=label_seed)
    return dict(n_pi=rep.n_pi, fanins=rep.fanins.copy(), pos=rep.pos.copy(),
                prob=popcount_prob(V).astype(np.float32), lev=levels(rep))


def build_cones(srcs, n_per, rng, lo=40, hi=400, min_pi=4, max_pi=64):
    out = []
    for s in srcs:
        rep = from_aig(read_aiger_into_aig(f'{B}/{s}.aig'))
        base = 1 + rep.n_pi
        got, tries = 0, 0
        while got < n_per and tries < n_per * 20:
            tries += 1
            root = int(rng.integers(base, rep.n_nodes))
            c = cone(rep, root, int(rng.integers(lo, hi)), rng)
            if c.n_gates < lo // 2 or not (min_pi <= c.n_pi <= max_pi):
                continue
            d = sample(c); d['src'] = s
            out.append(d); got += 1
        print(s, got, flush=True)
    return out


def rep_of(d):
    return Rep(d['n_pi'], d['fanins'], d['pos'])


SCRIPTS = {'S1': ['rw', 'rs', 'b', 'rf'], 'S2': ['b', 'rwz', 'rfz', 'rs2'], 'S3': ['rf', 'rw', 'rwz'],
           'S4': ['rs2', 'b', 'rw', 'rs']}


def variant(d, kind, seed):
    """Certified-equivalent variant. kind: S1..S4 (synthesis) or ('inj', rate)."""
    rep = rep_of(d)
    a = to_aig(rep)
    if isinstance(kind, tuple):
        b = inject(rep, kind[1], seed)
    else:
        b = synth(a, SCRIPTS[kind])
    ok = certify(a, b)
    rb = from_aig(b)
    return rb, ok


def match(d, rb, stats=None):
    """Cross-variant functional equivalence, SAT-verified.
    Candidates come from identical 4,096-pattern simulation signatures (up to complement); every
    reported pair (i, j, polarity) is then PROVEN equivalent by a SAT call on a shared-input miter.
    Signature matches that SAT refutes (e.g. rarely-active nodes) are discarded."""
    from sweep import Miter
    ra = rep_of(d)
    rng = np.random.default_rng(123)
    piw = rng.integers(0, 2**63, size=(ra.n_pi, NW), dtype=np.uint64) | \
        (rng.integers(0, 2, size=(ra.n_pi, NW), dtype=np.uint64) << np.uint64(63))
    Va = simulate(ra, NW, pi_words=piw)
    Vb = simulate(rb, NW, pi_words=piw)
    full = np.uint64(0xFFFFFFFFFFFFFFFF)
    canon = lambda row: (row ^ full) if (int(row[0]) & 1) else row
    buckets = {}
    for i in range(1 + ra.n_pi, ra.n_nodes):
        buckets.setdefault(canon(Va[i]).tobytes(), []).append(i)
    M = None
    I, J, P = [], [], []
    n_cand_nodes = n_refuted = 0
    for j in range(1 + rb.n_pi, rb.n_nodes):
        cand = buckets.get(canon(Vb[j]).tobytes())
        if not cand:
            continue
        n_cand_nodes += 1
        if M is None:
            M = Miter(ra, rb)
        for i in cand:
            p = int(not np.array_equal(Va[i], Vb[j]))
            eq, _ = M.check(i, j, p)
            if eq:
                I.append(i); J.append(j); P.append(p)
                break
            n_refuted += 1
    if stats is not None:
        stats['sig_nodes'] = stats.get('sig_nodes', 0) + n_cand_nodes
        stats['proven'] = stats.get('proven', 0) + len(I)
        stats['refuted_calls'] = stats.get('refuted_calls', 0) + n_refuted
    return np.array(I, dtype=np.int64), np.array(J, dtype=np.int64), np.array(P, dtype=np.int64)


def rep_sample(rb, src):
    d = sample(rb); d['src'] = src
    return d


KINDS = ['S1', 'S2', 'S3', 'S4', ('inj', 0.15), ('inj', 0.3)]
TKINDS = ['S1', 'S2', 'S3', 'S4', ('inj', 0.1), ('inj', 0.3), ('inj', 0.5)]


def _train_pair(args):
    n, d = args
    k = KINDS[n % len(KINDS)]
    rb, ok = variant(d, k, seed=n)
    I, J, P = match(d, rb)
    return dict(var=rep_sample(rb, d['src']), I=I, J=J, P=P, kind=str(k), cert=ok)


def _test_row(args):
    n, d = args
    row = {}
    for k in TKINDS:
        rb, ok = variant(d, k, seed=10_000 + n)
        I, J, P = match(d, rb)
        row[str(k)] = dict(var=rep_sample(rb, d['src']), I=I, J=J, P=P, cert=ok)
    return row


if __name__ == '__main__':
    import os
    from multiprocessing import Pool
    t = time.time()
    if not os.path.exists('cones.pkl'):
        rng = np.random.default_rng(0)
        train = build_cones(TRAIN_SRC, 250, rng)
        test = build_cones(TEST_SRC, 60, rng)
        pickle.dump(dict(train=train, test=test), open('cones.pkl', 'wb'))
    C = pickle.load(open('cones.pkl', 'rb')); train, test = C['train'], C['test']
    print('cones', len(train), len(test), f'{time.time()-t:.1f}s', flush=True)
    with Pool(2) as pool:
        if not os.path.exists('pairs.pkl'):
            pairs = []
            for i, r in enumerate(pool.imap(_train_pair, enumerate(train), chunksize=8)):
                pairs.append(r)
                if i % 250 == 0:
                    print('train pair', i, f'{time.time()-t:.1f}s', flush=True)
            pickle.dump(pairs, open('pairs.pkl', 'wb'))
        pairs = pickle.load(open('pairs.pkl', 'rb'))
        print('train pairs', len(pairs), 'cert ok', sum(p['cert'] for p in pairs), f'{time.time()-t:.1f}s', flush=True)
        if not os.path.exists('tvars.pkl'):
            tvars = []
            for i, r in enumerate(pool.imap(_test_row, enumerate(test), chunksize=4)):
                tvars.append(r)
                if i % 60 == 0:
                    print('test row', i, f'{time.time()-t:.1f}s', flush=True)
            pickle.dump(tvars, open('tvars.pkl', 'wb'))
        tvars = pickle.load(open('tvars.pkl', 'rb'))
    nfail = sum((not v['cert']) for r in tvars for v in r.values())
    print('test variants', len(tvars), 'cert failures', nfail, f'{time.time()-t:.1f}s', flush=True)
    pickle.dump(dict(train=train, test=test, pairs=pairs, tvars=tvars), open('data.pkl', 'wb'))
    print('done', flush=True)
