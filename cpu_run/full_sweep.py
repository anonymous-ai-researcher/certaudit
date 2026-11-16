"""SAT sweeping on full held-out EPFL circuits vs. their certified variants.
Usage: python full_sweep.py <model_tag> [circuits...]"""
import sys, os, json, pickle, time, numpy as np, torch
from aigverse.io import read_aiger_into_aig
from aiglib import from_aig, to_aig, synth, inject, certify
from data import B, sample, match, SCRIPTS
from model import AIGEncoder, collate
from sweep import run_pair

torch.set_num_threads(int(os.environ.get('THREADS', 1)))
CIRCS = ['random_control/ctrl', 'random_control/router', 'random_control/int2float', 'random_control/cavlc',
         'arithmetic/adder']
KINDS = ['S1', 'S3', ('inj', 0.1), ('inj', 0.3)]


def variants():
    path = 'full_variants.pkl'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    out = {}
    for c in CIRCS:
        a = read_aiger_into_aig(f'{B}/{c}.aig'); r = from_aig(a)
        d = sample(r); d['src'] = c
        out[c] = {'orig': d}
        for k in KINDS:
            b = synth(a, SCRIPTS[k]) if isinstance(k, str) else inject(r, k[1], 4242)
            ok = certify(a, b)
            rb = from_aig(b); s = sample(rb); s['src'] = c
            I, J, P = match(d, rb)
            out[c][str(k)] = dict(var=s, I=I, J=J, P=P, cert=ok)
            print(c, k, 'gates', len(d['fanins']), '->', len(s['fanins']), 'cert', ok, flush=True)
    pickle.dump(out, open(path, 'wb'))
    return out


if __name__ == '__main__':
    tag = sys.argv[1]
    V = variants()
    m = AIGEncoder(64, 2, proj='_ph' in tag)
    m.load_state_dict(torch.load(f'{tag}.pt')); m.eval()

    def emb(s):
        with torch.no_grad():
            t = time.perf_counter(); h, _ = m(collate([s])); return m.z(h).numpy(), time.perf_counter() - t
    res = []
    for c in CIRCS:
        eo, to = emb(V[c]['orig'])
        for k in KINDS:
            v = V[c][str(k)]
            ev, tv = emb(v['var'])
            for bits in (4, 8, 16, 32, 64):
                for pol in ('random', 'topo', 'level', 'emb', 'oracle'):
                    seeds = (0, 1, 2) if pol == 'random' else (0,)
                    for sd in seeds:
                        r = run_pair(V[c]['orig'], v['var'], v['I'], v['J'], v['P'], eo, ev, bits=bits, policy=pol, seed=sd)
                        r.update(circ=c, kind=str(k), bits=bits, policy=pol, seed=sd, emb_time=to + tv, model=tag)
                        res.append(r)
                pass
    json.dump(res, open(f'sweep2_{tag}.json', 'w'))
