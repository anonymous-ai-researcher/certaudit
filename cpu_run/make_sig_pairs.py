"""Experiment A input: training pairs labelled by simulation alone, no SAT.

Identical to data.match() except that the first signature candidate (topological order) is
accepted without a proof. Everything else is unchanged: same variants, same canonical
signature up to complement, same shared input patterns for both circuits. Written as
pairs_sig{W}.pkl, aligned index-for-index with D['pairs'], so training differs from EqAlign
only in where the positive pairs come from."""
import os, sys, pickle
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np
from aiglib import simulate
from data import rep_of

D = pickle.load(open('data.pkl', 'rb'))
full = np.uint64(0xFFFFFFFFFFFFFFFF)


def sig_match(d, rb, nw):
    ra = rep_of(d)
    rng = np.random.default_rng(123)                           # same pattern seed as data.match
    piw = rng.integers(0, 2**63, size=(ra.n_pi, nw), dtype=np.uint64) | \
        (rng.integers(0, 2, size=(ra.n_pi, nw), dtype=np.uint64) << np.uint64(63))
    Va, Vb = simulate(ra, nw, pi_words=piw), simulate(rb, nw, pi_words=piw)
    canon = lambda row: (row ^ full) if (int(row[0]) & 1) else row
    buckets = {}
    for i in range(1 + ra.n_pi, ra.n_nodes):
        buckets.setdefault(canon(Va[i]).tobytes(), []).append(i)
    I, J, P = [], [], []
    for j in range(1 + rb.n_pi, rb.n_nodes):
        cand = buckets.get(canon(Vb[j]).tobytes())
        if cand:
            i = cand[0]
            I.append(i); J.append(j); P.append(int(not np.array_equal(Va[i], Vb[j])))
    return np.array(I, np.int64), np.array(J, np.int64), np.array(P, np.int64)


if __name__ == '__main__':
    for bits in (64, 4096):
        nw = bits // 64
        out, n_pair, n_agree, n_proven = [], 0, 0, 0
        for d, pr in zip(D['train'], D['pairs']):
            I, J, P = sig_match(d, rep_of(pr['var']), nw)
            proven = {(int(j), int(i), int(p)) for i, j, p in zip(pr['I'], pr['J'], pr['P'])}
            n_pair += len(I); n_proven += len(proven)
            n_agree += sum((int(j), int(i), int(p)) in proven for i, j, p in zip(I, J, P))
            out.append(dict(I=I, J=J, P=P))
        pickle.dump(out, open(f'pairs_sig{bits}.pkl', 'wb'))
        wrong = n_pair - n_agree
        print(f'{bits:5d}-bit signatures: {n_pair} pairs, {n_agree} identical to a SAT-proven pair, '
              f'{wrong} not ({100 * wrong / max(n_pair, 1):.1f}%); SAT-proven set has {n_proven}', flush=True)
