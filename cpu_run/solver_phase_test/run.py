"""E19 solves (rules: _.md). One process per design, resumable: phase_exp/res/<design>.jsonl.
  python phase_exp/run.py <design>"""
import os, sys, json, pickle, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build, collate
from data import rep_of
from phase_exp.common import cnf_miter, solve, phase_lits, rule, budget

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
REG = {'Base': 'base', 'Aug': 'aug', 'TT': 'ttsim_ph_l0.1', 'Circ': 'circ_ph_l0.2', 'Ours': 'eqalign_ph_l0.2'}
SEEDS = [0, 1, 2]; TAUS = [0.01, 0.5]; PLACEBO = ('Base', 'Ours')


def preds(tag, s):
    m = build(tag); m.load_state_dict(torch.load(os.path.join(ROOT, f'v_{tag}.pt'))); m.eval()
    with torch.no_grad():
        _, p = m(collate([s]))
    return p.numpy().astype(np.float64)


def main(design, only=None):
    inst = pickle.load(open(os.path.join(HERE, 'inst', design.replace('/', '__') + '.pkl'), 'rb'))
    os.makedirs(os.path.join(HERE, 'res'), exist_ok=True)
    stem = design.replace('/', '__')
    out = os.path.join(HERE, 'res', stem + ('' if only is None else '.part' + '_'.join(map(str, only))) + '.jsonl')
    done = set()
    import glob as _g
    for fn in _g.glob(os.path.join(HERE, 'res', stem + '*.jsonl')):      # all parts of this design
        for l in open(fn):
            r = json.loads(l); done.add((r['mut'], r['X'], r['src'], r['seed'], r['tau'], r['mode']))
    f = open(out, 'a'); B = budget(design)
    impl, muts = inst['impl'], inst['muts']
    # predictions: model (regime, seed) x circuit; oracle = simulated probability
    circ = {X: v['s'] for X, v in impl.items()}
    for k, m in enumerate(muts):
        circ[f'M{k}'] = m['s']
    P = {('oracle', 0, c): np.asarray(s['prob'], dtype=np.float64) for c, s in circ.items()}
    cdir = os.path.join(HERE, 'predcache', design.replace('/', '__')); os.makedirs(cdir, exist_ok=True)
    need = set(impl) | {f'M{k}' for k in range(len(muts)) if only is None or k in only}
    for reg, pre in REG.items():
        for sd in SEEDS:
            for c, s in circ.items():
                if c not in need:
                    continue
                fn = os.path.join(cdir, f'{pre}_s{sd}__{c}.npy')   # deterministic predictions, cached across restarts
                if os.path.exists(fn):
                    P[(reg, sd, c)] = np.load(fn)
                else:
                    P[(reg, sd, c)] = preds(f'{pre}_s{sd}', s); np.save(fn, P[(reg, sd, c)])
    reps = {c: rep_of(s) for c, s in circ.items()}
    rng = np.random.default_rng(2026)

    def run(key, cls, va, vb, hX, hM, changed=None):
        if key in done:
            return
        ph = None if hX is None else phase_lits(va, hX) + phase_lits(vb, hM)
        st, conf, sec = solve(cls, phases=ph, budget=B)
        r = dict(design=design, mut=key[0], X=key[1], src=key[2], seed=key[3], tau=key[4], mode=key[5],
                 status={True: 'SAT', False: 'UNSAT', None: 'BUDGET'}[st], conf=int(conf), sec=round(sec, 4),
                 changed=changed, budget=B)
        f.write(json.dumps(r) + '\n'); f.flush()

    for k in range(len(muts)):
        M = f'M{k}'
        if only is not None and k not in only:
            # keep the random stream identical to a full sequential run
            for X in impl:
                for tau in TAUS:
                    hX, hM = rule(P[('oracle', 0, X)], tau), rule(P[('oracle', 0, M)], tau)
                    rng.integers(0, 2, len(hX)); rng.integers(0, 2, len(hM))
                    if X != 'D':
                        for reg in PLACEBO:
                            for sd in SEEDS:
                                hX = rule(P[(reg, sd, X)], tau); hD = rule(P[(reg, sd, 'D')], tau)
                                I, J, Pp = impl[X]['I'], impl[X]['J'], impl[X]['P']
                                hS = hX.copy(); src = hD[I]; hS[J] = np.where(src >= 0, src ^ Pp, -1)
                                ch = int((hS != hX).sum()); n_pi = circ[X]['n_pi']
                                rng.choice(np.arange(1 + n_pi, len(hX)), size=min(ch, len(hX) - 1 - n_pi), replace=False)
                                rng.integers(1 + circ['D']['n_pi'], len(hD), size=min(ch, len(hX) - 1 - n_pi))
            continue
        for X in impl:
            cls, va, vb = cnf_miter(reps[X], reps[M])
            run((k, X, 'default', 0, 0, 'own'), cls, va, vb, None, None)
            for tau in TAUS:
                hX, hM = rule(P[('oracle', 0, X)], tau), rule(P[('oracle', 0, M)], tau)
                run((k, X, 'oracle', 0, tau, 'own'), cls, va, vb, hX, hM)
                rX = np.where(hX >= 0, rng.integers(0, 2, len(hX)), -1) if tau < 0.5 else rng.integers(0, 2, len(hX))
                rM = np.where(hM >= 0, rng.integers(0, 2, len(hM)), -1) if tau < 0.5 else rng.integers(0, 2, len(hM))
                run((k, X, 'random', 0, tau, 'own'), cls, va, vb, rX, rM)
                for reg in REG:
                    for sd in SEEDS:
                        hX, hM = rule(P[(reg, sd, X)], tau), rule(P[(reg, sd, M)], tau)
                        run((k, X, reg, sd, tau, 'own'), cls, va, vb, hX, hM)
                        if X == 'D':
                            continue
                        hD = rule(P[(reg, sd, 'D')], tau); I, J, Pp = impl[X]['I'], impl[X]['J'], impl[X]['P']
                        hS = hX.copy(); src = hD[I]; hS[J] = np.where(src >= 0, src ^ Pp, -1)
                        ch = int((hS != hX).sum())
                        run((k, X, reg, sd, tau, 'swap'), cls, va, vb, hS, hM, ch)
                        if reg in PLACEBO:
                            nX = len(hX); n_pi = circ[X]['n_pi']
                            gates = np.arange(1 + n_pi, nX)
                            idx = rng.choice(gates, size=min(ch, len(gates)), replace=False)
                            dsrc = rng.integers(1 + circ['D']['n_pi'], len(hD), size=len(idx))
                            hP = hX.copy(); hP[idx] = hD[dsrc]
                            run((k, X, reg, sd, tau, 'placebo'), cls, va, vb, hP, hM, int((hP != hX).sum()))
        print(design, 'mutant', k, 'done', flush=True)
    f.close()


if __name__ == '__main__':
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else None)
