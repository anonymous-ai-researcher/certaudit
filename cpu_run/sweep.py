"""Downstream: learned candidate ordering for SAT sweeping across an original/variant pair.

For every variant gate j (topological order) we look for an equivalent (or complementary) original
gate i, as in fraiging / combinational equivalence checking. Candidates are the original gates whose
k-pattern simulation signature matches j (up to complement), refined on the fly by SAT counterexamples
(counterexample-guided refinement, as in ABC fraig). Candidates are tried in a given order until one is
proven equivalent (UNSAT) or all are refuted. Ordering policies:
  random   : uniformly random order
  level    : ascending |level(i) - level(j)|, ties by index (non-learned structural heuristic)
  emb      : descending cosine similarity of learned node embeddings
  oracle   : true equivalent first (lower bound)
Every equivalence is decided by a real SAT call (CaDiCaL via PySAT), so proofs double as a check that
the 4,096-pattern signature matches used in the audit are genuine equivalences."""
import time, numpy as np
from pysat.solvers import Cadical153
from aiglib import simulate, levels
from data import rep_of, NW


class Miter:
    """Both circuits Tseitin-encoded in one incremental solver over shared PI variables."""

    def __init__(self, ra, rb):
        self.s = Cadical153()
        self.na, self.nb = ra.n_nodes, rb.n_nodes
        nv = [0]

        def new():
            nv[0] += 1; return nv[0]
        const = new(); self.s.add_clause([-const])
        pis = [new() for _ in range(ra.n_pi)]
        self.va = self._enc(ra, const, pis, new)
        self.vb = self._enc(rb, const, pis, new)
        self.new = new

    def _enc(self, r, const, pis, new):
        v = [const] + pis
        base = 1 + r.n_pi
        for g, (a, b) in enumerate(r.fanins):
            x = v[int(a) >> 1] * (-1 if int(a) & 1 else 1)
            y = v[int(b) >> 1] * (-1 if int(b) & 1 else 1)
            o = new(); v.append(o)
            self.s.add_clause([-o, x]); self.s.add_clause([-o, y]); self.s.add_clause([o, -x, -y])
        return v

    def check(self, i, j, compl):
        """True if orig node i == (compl ? ~var node j : var node j). Returns (equiv, model or None)."""
        a, b = self.va[i], self.vb[j] * (-1 if compl else 1)
        t = self.new()
        # t -> (a xor b)
        self.s.add_clause([-t, a, b]); self.s.add_clause([-t, -a, -b])
        sat = self.s.solve(assumptions=[t])
        if sat:
            m = self.s.get_model()
            self.s.add_clause([-t])
            return False, m
        self.s.add_clause([-t])
        return True, None


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
