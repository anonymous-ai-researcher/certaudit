"""AIG utilities: conversion from aigverse, bit-parallel simulation, levelization,
function-preserving transformations (synthesis scripts and identity injection),
and SAT-based equivalence certification."""
import numpy as np
from aigverse.networks import Aig
from aigverse import algorithms as A


class Rep:
    """Array representation. Nodes: 0 = const0, 1..n_pi = PIs, then AND gates in topo order.
    Literal = 2*node + complement."""

    def __init__(self, n_pi, fanins, pos):
        self.n_pi = n_pi
        self.fanins = np.asarray(fanins, dtype=np.int64).reshape(-1, 2)  # per gate
        self.pos = np.asarray(pos, dtype=np.int64)
        self.n_nodes = 1 + n_pi + len(self.fanins)

    @property
    def n_gates(self):
        return len(self.fanins)


def from_aig(aig):
    idx = {}
    idx[0] = 0
    for k, p in enumerate(aig.pis()):
        idx[aig.node_to_index(p)] = 1 + k
    n_pi = aig.num_pis
    fan = []
    nxt = 1 + n_pi
    for g in aig.gates():
        gi = aig.node_to_index(g)
        lits = []
        for s in aig.fanins(g):
            lits.append(2 * idx[s.index] + int(s.complement))
        fan.append(lits)
        idx[gi] = nxt
        nxt += 1
    pos = [2 * idx[s.index] + int(s.complement) for s in aig.pos()]
    return Rep(n_pi, fan, pos)


def to_aig(rep):
    aig = Aig()
    sig = [aig.get_constant(False)]
    for _ in range(rep.n_pi):
        sig.append(aig.create_pi())

    def lit(l):
        s = sig[l >> 1]
        return aig.create_not(s) if (l & 1) else s

    for a, b in rep.fanins:
        sig.append(aig.create_and(lit(int(a)), lit(int(b))))
    for l in rep.pos:
        aig.create_po(lit(int(l)))
    return aig


def simulate(rep, n_words=64, seed=0, pi_words=None):
    """Bit-parallel random simulation. Returns uint64 array [n_nodes, n_words]."""
    rng = np.random.default_rng(seed)
    V = np.zeros((rep.n_nodes, n_words), dtype=np.uint64)
    if pi_words is None:
        pi_words = rng.integers(0, 2**63, size=(rep.n_pi, n_words), dtype=np.uint64) ^ \
            (rng.integers(0, 2, size=(rep.n_pi, n_words), dtype=np.uint64) << np.uint64(63))
    V[1:1 + rep.n_pi] = pi_words
    full = np.uint64(0xFFFFFFFFFFFFFFFF)
    base = 1 + rep.n_pi
    f = rep.fanins
    for g in range(rep.n_gates):
        a, b = f[g]
        va = V[a >> 1] ^ (full if a & 1 else np.uint64(0))
        vb = V[b >> 1] ^ (full if b & 1 else np.uint64(0))
        V[base + g] = va & vb
    return V


def popcount_prob(V):
    # fraction of ones per row
    b = np.unpackbits(V.view(np.uint8), axis=1)
    return b.mean(axis=1)


def levels(rep):
    L = np.zeros(rep.n_nodes, dtype=np.int64)
    base = 1 + rep.n_pi
    for g, (a, b) in enumerate(rep.fanins):
        L[base + g] = 1 + max(L[a >> 1], L[b >> 1])
    return L


# ---------------- transformations ----------------

def synth(aig, script, seed=0):
    """Apply a list of mockturtle passes (all function preserving)."""
    a = aig.clone()
    for step in script:
        if step == "rw":
            a = A.aig_cut_rewriting(a)
        elif step == "rwz":
            a = A.aig_cut_rewriting(a, allow_zero_gain=True)
        elif step == "rs":
            A.aig_resubstitution(a, inplace=True)
        elif step == "rs2":
            A.aig_resubstitution(a, max_pis=10, max_inserts=3, inplace=True)
        elif step == "b":
            a = A.balancing(a)
        elif step == "rf":
            A.sop_refactoring(a, inplace=True)
        elif step == "rfz":
            A.sop_refactoring(a, allow_zero_gain=True, inplace=True)
        # ---- held-out passes (never used for training) ----
        elif step == "rw6dc":
            a = A.aig_cut_rewriting(a, cut_size=4, use_dont_cares=True, allow_zero_gain=True)
        elif step == "rsdc":
            A.aig_resubstitution(a, use_dont_cares=True, window_size=12, max_pis=10, inplace=True)
        elif step == "besop":
            a = A.balancing(a, rebalance_function='esop')
        elif step == "rfdc":
            A.sop_refactoring(a, use_dont_cares=True, use_reconvergence_cut=True, inplace=True)
        a = A.cleanup_dangling(a)
    return a


def inject(rep, rate, seed, kinds=("dup", "shannon", "absorb")):
    """Function-preserving identity injection. Each selected gate g = x & y is rebuilt as an
    equivalent but structurally different sub-network:
      dup:     (x & y) & ~(~x & ~y)                     [(x&y)&(x|y)]
      shannon: ((x&y) & z) | ((x&y) & ~z)              [Shannon expansion on random earlier z]
      absorb:  (x & y) & ~((x & y) & ~x)  ...          [x&y & ~(x&y&~x) = x&y]
    Returns new Rep. Gate ordering stays topological."""
    rng = np.random.default_rng(seed)
    aig = Aig()
    sig = [aig.get_constant(False)]
    for _ in range(rep.n_pi):
        sig.append(aig.create_pi())

    def lit(l):
        s = sig[l >> 1]
        return aig.create_not(s) if (l & 1) else s

    for g, (a, b) in enumerate(rep.fanins):
        x, y = lit(int(a)), lit(int(b))
        core = aig.create_and(x, y)
        if rng.random() < rate:
            k = kinds[rng.integers(len(kinds))]
            if k == "dup":
                orxy = aig.create_not(aig.create_and(aig.create_not(x), aig.create_not(y)))
                out = aig.create_and(core, orxy)
            elif k == "shannon":
                zi = int(rng.integers(1, len(sig)))
                z = sig[zi]
                t1 = aig.create_and(core, z)
                t0 = aig.create_and(core, aig.create_not(z))
                out = aig.create_not(aig.create_and(aig.create_not(t1), aig.create_not(t0)))
            elif k == "absorb":
                junk = aig.create_and(core, aig.create_not(x))  # == 0 functionally
                out = aig.create_and(core, aig.create_not(junk))
            elif k == "xor":   # held-out identity: x & y == (x ^ y) ^ (x | y)
                def X(a, b):
                    return aig.create_not(aig.create_and(aig.create_not(aig.create_and(a, aig.create_not(b))),
                                                         aig.create_not(aig.create_and(aig.create_not(a), b))))
                orxy = aig.create_not(aig.create_and(aig.create_not(x), aig.create_not(y)))
                out = X(X(x, y), orxy)
            else:
                raise ValueError(k)
            sig.append(out)
        else:
            sig.append(core)
    for l in rep.pos:
        aig.create_po(lit(int(l)))
    aig = A.cleanup_dangling(aig)
    return aig


def certify(aig1, aig2, conflict_limit=0):
    return A.equivalence_checking(aig1, aig2, conflict_limit=conflict_limit)
