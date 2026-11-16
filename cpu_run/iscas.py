"""Second benchmark suite (test only): ISCAS85 and the combinational cores of ISCAS89 (flip-flop outputs
become inputs, flip-flop inputs become outputs). Builds circuits (fanin cones) with 7 fixed + 2 held-out
certified variants and SAT-proven matches, exactly as for the EPFL test set (no adversarial variants)."""
import re, os, sys, pickle, time, subprocess, numpy as np
from aigverse.networks import Aig

ROOT = '${WORK}/bench2/hdl'
DESIGNS = ['iscas85/bench/' + d for d in ['c432', 'c499', 'c880', 'c1355', 'c1908', 'c2670', 'c3540', 'c5315', 'c6288', 'c7552']] + \
          ['iscas89/bench/' + d for d in ['s5378', 's9234', 's13207', 's15850', 's35932', 's38417', 's38584']]


def read_bench(path):
    ins, outs, defs = [], [], {}
    for line in open(path):
        line = line.split('#')[0].strip()
        if not line:
            continue
        m = re.match(r'INPUT\((.+)\)', line)
        if m:
            ins.append(m.group(1).strip()); continue
        m = re.match(r'OUTPUT\((.+)\)', line)
        if m:
            outs.append(m.group(1).strip()); continue
        m = re.match(r'(\S+)\s*=\s*(\w+)\((.*)\)', line)
        if m:
            defs[m.group(1)] = (m.group(2).upper(), [x.strip() for x in m.group(3).split(',') if x.strip()])
    # combinational core: DFF outputs are pseudo-inputs, DFF inputs pseudo-outputs
    for k, (t, a) in list(defs.items()):
        if t == 'DFF':
            ins.append(k); outs.append(a[0]); del defs[k]
    aig = Aig(); sig = {}
    for i in ins:
        sig[i] = aig.create_pi()

    def get(n, stack=()):
        if n in sig:
            return sig[n]
        t, a = defs[n]
        xs = [get(x) for x in a]
        if t in ('BUFF', 'BUF'):
            r = xs[0]
        elif t == 'NOT':
            r = aig.create_not(xs[0])
        elif t in ('AND', 'NAND'):
            r = aig.create_nary_and(xs) if len(xs) > 2 else aig.create_and(xs[0], xs[1]) if len(xs) == 2 else xs[0]
            r = aig.create_not(r) if t == 'NAND' else r
        elif t in ('OR', 'NOR'):
            r = aig.create_nary_or(xs) if len(xs) > 2 else aig.create_or(xs[0], xs[1]) if len(xs) == 2 else xs[0]
            r = aig.create_not(r) if t == 'NOR' else r
        elif t in ('XOR', 'XNOR'):
            r = xs[0]
            for x in xs[1:]:
                r = aig.create_xor(r, x)
            r = aig.create_not(r) if t == 'XNOR' else r
        else:
            raise ValueError(t)
        sig[n] = r
        return r
    sys.setrecursionlimit(100000)
    for o in outs:
        aig.create_po(get(o))
    return aig


if __name__ == '__main__':
    from aiglib import from_aig
    from data import cone, sample
    rng = np.random.default_rng(2027)
    test = []
    t0 = time.time()
    for d in DESIGNS:
        rep = from_aig(read_bench(f'{ROOT}/{d}.bench'))
        base = 1 + rep.n_pi; got, tries = 0, 0
        while got < 40 and tries < 800:
            tries += 1
            c = cone(rep, int(rng.integers(base, rep.n_nodes)), int(rng.integers(40, 400)), rng)
            if c.n_gates < 20 or not (4 <= c.n_pi <= 64):
                continue
            s = sample(c); s['src'] = d.split('/')[-1]; test.append(s); got += 1
        print(d, rep.n_pi, rep.n_gates, 'cones', got, f'{time.time()-t0:.0f}s', flush=True)
    pickle.dump(test, open('iscas_cones.pkl', 'wb'))
    print('ISCAS cones', len(test))
