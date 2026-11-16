"""E19 shared pieces: miter CNF, phase hints, one CaDiCaL solve."""
import time, numpy as np
from pysat.solvers import Cadical153

ABC = 'work/ext/abc/abc'
SMALL = {'arithmetic/adder', 'random_control/cavlc', 'random_control/ctrl', 'random_control/int2float', 'random_control/router'}


def budget(design):
    return 200000 if design in SMALL else 100000


def _enc(r, cls, const, pis, new):
    v = [const] + pis
    for a, b in r.fanins:
        x = v[int(a) >> 1] * (-1 if int(a) & 1 else 1); y = v[int(b) >> 1] * (-1 if int(b) & 1 else 1)
        o = new(); v.append(o); cls += [[-o, x], [-o, y], [o, -x, -y]]
    return v


def cnf_miter(ra, rb):
    """Shared-PI miter: some output pair differs. Returns (clauses, va, vb): va[v] = CNF var of node v of ra."""
    cls, n = [], [0]

    def new():
        n[0] += 1; return n[0]
    c = new(); cls.append([-c]); pis = [new() for _ in range(ra.n_pi)]
    va = _enc(ra, cls, c, pis, new); vb = _enc(rb, cls, c, pis, new); xs = []
    lit = lambda v, l: v[int(l) >> 1] * (-1 if int(l) & 1 else 1)
    for la, lb in zip(ra.pos, rb.pos):
        a, b = lit(va, la), lit(vb, lb); t = new(); xs.append(t)
        cls += [[-t, a, b], [-t, -a, -b], [t, -a, b], [t, a, -b]]
    cls.append(xs)
    return cls, va, vb


def phase_lits(vmap, hint):
    """hint[v] in {0, 1, -1 (none)} per node v of a circuit (v >= 1) -> phase literals on its CNF vars."""
    return [(vmap[v] if h == 1 else -vmap[v]) for v, h in enumerate(hint) if v >= 1 and h >= 0]


def rule(p, tau):
    p = np.asarray(p, dtype=np.float64)
    if tau >= 0.5:
        return np.where(p > 0.5, 1, np.where(p < 0.5, 0, -1))
    return np.where(p < tau, 0, np.where(p > 1 - tau, 1, -1))


def solve(cls, va=None, vb=None, phases=None, budget=200000):
    s = Cadical153(bootstrap_with=cls)
    if phases:
        s.set_phases(phases)
    s.conf_budget(budget)
    t = time.perf_counter(); st = s.solve_limited(); sec = time.perf_counter() - t
    conf = s.accum_stats()['conflicts']; s.delete()
    return st, conf, sec


def fast_match(d, rb, conf_per_call=5000):
    """SAT-proven gate correspondences, as data.match (same 4,096-pattern candidates, same proof per pair), but sweeping
    style: every proven equivalence is added to the solver (sound, it is proven) and each call has a conflict budget;
    a pair whose proof exceeds the budget is simply not reported. Used for the large designs (log2, sqrt)."""
    from data import rep_of, NW
    from aiglib import simulate
    ra = rep_of(d)
    rng = np.random.default_rng(123)
    piw = rng.integers(0, 2**63, size=(ra.n_pi, NW), dtype=np.uint64) | \
        (rng.integers(0, 2, size=(ra.n_pi, NW), dtype=np.uint64) << np.uint64(63))
    Va = simulate(ra, NW, pi_words=piw); Vb = simulate(rb, NW, pi_words=piw)
    full = np.uint64(0xFFFFFFFFFFFFFFFF)
    canon = lambda row: (row ^ full) if (int(row[0]) & 1) else row
    buckets = {}
    for i in range(1 + ra.n_pi, ra.n_nodes):
        buckets.setdefault(canon(Va[i]).tobytes(), []).append(i)
    cls, n = [], [0]

    def new():
        n[0] += 1; return n[0]
    c = new(); cls.append([-c]); pis = [new() for _ in range(ra.n_pi)]
    va = _enc(ra, cls, c, pis, new); vb = _enc(rb, cls, c, pis, new)
    s = Cadical153(bootstrap_with=cls); top = [n[0]]
    I, J, P = [], [], []; stats = dict(sig_nodes=0, refuted=0, budget=0)
    for j in range(1 + rb.n_pi, rb.n_nodes):
        cand = buckets.get(canon(Vb[j]).tobytes())
        if not cand:
            continue
        stats['sig_nodes'] += 1
        for i in cand:
            p = int(not np.array_equal(Va[i], Vb[j]))
            a, b = va[i], vb[j] * (-1 if p else 1)
            top[0] += 1; t = top[0]
            s.add_clause([-t, a, b]); s.add_clause([-t, -a, -b])
            s.conf_budget(conf_per_call); r = s.solve_limited(assumptions=[t])
            s.add_clause([-t])
            if r is False:
                I.append(i); J.append(j); P.append(p)
                s.add_clause([-a, b]); s.add_clause([a, -b])      # proven: merge them
                break
            if r is None:
                stats['budget'] += 1
            else:
                stats['refuted'] += 1
    s.delete()
    return np.array(I, dtype=np.int64), np.array(J, dtype=np.int64), np.array(P, dtype=np.int64), stats
