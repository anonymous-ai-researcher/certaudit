"""E19 instance builder (rules: _.md). For each EPFL test design: implementations D, S1 (mockturtle, the
paper's S1 script), R (ABC resyn2); certification (ABC cec for all, mockturtle too on the small designs); SAT-proven
gate correspondences D -> S1, D -> R (data.match); hard single-edge mutants M of D (undetected by 1,024 random
patterns, SAT within the default-solver budget). Writes phase_exp/inst/<design>.pkl.
  python phase_exp/build.py <design>"""
import os, sys, copy, pickle, subprocess, tempfile, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aigverse.io import read_aiger_into_aig, write_aiger
from aiglib import from_aig, to_aig, synth, certify, simulate
from data import B, sample, match, SCRIPTS
from phase_exp.common import cnf_miter, solve, ABC, SMALL, budget, fast_match

RESYN2 = 'balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance'
N_MUT, MAX_DRAW = 6, 3000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inst')


def abc(cmd):
    r = subprocess.run([ABC, '-q', cmd], capture_output=True, text=True, timeout=3600)
    return r.stdout + r.stderr


def detected(r, m, words=16, seed=7):
    rng = np.random.default_rng(seed)
    piw = rng.integers(0, 2**63, size=(r.n_pi, words), dtype=np.uint64) | \
        (rng.integers(0, 2, size=(r.n_pi, words), dtype=np.uint64) << np.uint64(63))
    Va, Vb = simulate(r, words, pi_words=piw), simulate(m, words, pi_words=piw)
    full = np.uint64(0xFFFFFFFFFFFFFFFF)
    for la, lb in zip(r.pos, m.pos):
        x = Va[int(la) >> 1] ^ (full if int(la) & 1 else np.uint64(0))
        y = Vb[int(lb) >> 1] ^ (full if int(lb) & 1 else np.uint64(0))
        if (x != y).any():
            return True
    return False


def main(design):
    os.makedirs(OUT, exist_ok=True); out = os.path.join(OUT, design.replace('/', '__') + '.pkl')
    if os.path.exists(out):
        print('exists', out); return
    t0 = time.time(); log = {}
    a = read_aiger_into_aig(f'{B}/{design}.aig'); rD = from_aig(a); D = sample(rD); D['src'] = design
    impl = {'D': dict(s=D)}
    tmp = tempfile.mkdtemp(); fa = os.path.join(tmp, 'a.aig'); write_aiger(to_aig(rD), fa)
    # S1: mockturtle, the paper's script
    bS = synth(a, SCRIPTS['S1']); rS = from_aig(bS)
    # R: ABC resyn2
    fr = os.path.join(tmp, 'r.aig'); abc(f'read_aiger {fa}; strash; {RESYN2}; strash; write_aiger {fr}')
    bR = read_aiger_into_aig(fr); rR = from_aig(bR)
    for name, b, rb in (('S1', bS, rS), ('R', bR, rR)):
        fb = os.path.join(tmp, name + '.aig'); write_aiger(b, fb)
        ok_abc = 'Networks are equivalent' in abc(f'cec -n {fa} {fb}')
        ok_mt = bool(certify(a, b)) if design in SMALL else None
        if not ok_abc or ok_mt is False:
            print(design, name, 'NOT certified', ok_abc, ok_mt); log[name] = 'not certified'; continue
        s = sample(rb); s['src'] = design
        t = time.time(); I, J, P, st = fast_match(D, rb)   # identical to data.match on the small designs (checked)
        impl[name] = dict(s=s, I=I, J=J, P=P, cert_abc=ok_abc, cert_mt=ok_mt, match_sec=time.time() - t, match_stats=st)
        print(design, name, 'gates', len(D['fanins']), '->', len(s['fanins']), 'proven', len(I), f'{time.time() - t:.0f}s', flush=True)
    # hard mutants
    rng = np.random.default_rng(sum(map(ord, design)))
    muts, draws, stats = [], 0, dict(detected=0, unsat=0, unknown=0)
    F0 = np.array(rD.fanins)
    seen = set()
    while len(muts) < N_MUT and draws < MAX_DRAW and len(seen) < 2 * len(F0):
        g, side = int(rng.integers(len(F0))), int(rng.integers(2))
        if (g, side) in seen:
            continue                                    # each mutation drawn at most once
        seen.add((g, side)); draws += 1
        m = copy.deepcopy(rD); F = F0.copy(); F[g, side] ^= 1; m.fanins = F
        if detected(rD, m):
            stats['detected'] += 1; continue
        st, conf, sec = solve(*cnf_miter(rD, m), phases=None, budget=budget(design))
        if st is None:
            stats['unknown'] += 1; continue
        if st is False:
            stats['unsat'] += 1; continue
        Ms = sample(m); Ms['src'] = design
        muts.append(dict(gate=g, side=side, s=Ms, default_conf_D=conf)); print(design, 'mutant', len(muts), 'gate', g, 'conf', conf, flush=True)
    pickle.dump(dict(design=design, impl=impl, muts=muts, draws=draws, mut_stats=stats, log=log, sec=time.time() - t0), open(out, 'wb'))
    print(design, 'done', len(muts), 'mutants from', draws, 'draws', stats, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1])
