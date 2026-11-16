"""Evaluate trained models: accuracy, cross-variant inconsistency on certified-equivalent nodes,
equivalent-node retrieval from embeddings, and worst-case error under certified adversarial rewrites."""
import sys, os, json, pickle, time, numpy as np, torch, torch.nn.functional as F
from model import AIGEncoder, collate
from data import rep_of, rep_sample, match, SCRIPTS
from aiglib import to_aig, from_aig, synth, inject, certify

torch.set_num_threads(int(os.environ.get("THREADS", 2)))
D = pickle.load(open('data.pkl', 'rb'))
test, tvars = D['test'], D['tvars']
KINDS = list(tvars[0].keys())


@torch.no_grad()
def run(model, s):
    b = collate([s])
    h, p = model(b)
    return model.z(h).numpy(), p.numpy()


def gates(s):
    return np.arange(1 + s['n_pi'], 1 + s['n_pi'] + len(s['fanins']))


def adv_variants(d, n, K=16, max_growth=0.30):
    """Random search space of certified-equivalent rewrites with bounded size growth."""
    rng = np.random.default_rng(777 + n)
    rep = rep_of(d); a = to_aig(rep); out = []
    steps = ['rw', 'rwz', 'rs', 'rs2', 'b', 'rf', 'rfz']
    tries = 0
    while len(out) < K and tries < 4 * K:
        tries += 1
        if rng.random() < 0.5:
            sc = list(rng.choice(steps, size=int(rng.integers(1, 5))))
            b = synth(a, sc)
            # optionally inject on top
            if rng.random() < 0.5:
                b = inject(from_aig(b), float(rng.uniform(0.02, 0.15)), int(rng.integers(1 << 30)))
        else:
            b = inject(rep, float(rng.uniform(0.02, 0.2)), int(rng.integers(1 << 30)),
                       kinds=tuple(rng.choice(['dup', 'shannon', 'absorb'], size=int(rng.integers(1, 4)), replace=False)))
        if b.num_gates > (1 + max_growth) * rep.n_gates:
            continue
        if not certify(a, b):
            continue
        rb = from_aig(b)
        I, J, P = match(d, rb)
        out.append(dict(var=rep_sample(rb, d['src']), I=I, J=J, P=P))
    return out


def shash(s):
    """Structural hash of each node's fanin cone (PIs keyed by index, commutative AND)."""
    n = 1 + s['n_pi'] + len(s['fanins'])
    H = [hash(('c',))] + [hash(('pi', k)) for k in range(s['n_pi'])]
    for a, b in s['fanins']:
        x = (H[int(a) >> 1], int(a) & 1); y = (H[int(b) >> 1], int(b) & 1)
        H.append(hash(('and',) + tuple(sorted([x, y]))))
    return np.array(H, dtype=object)


def evaluate(model):
    R = {'incons_rs': [], 'err_rs': [], 'top1_rs': [], 'frac_rs': [], 'mae_orig': [], 'mae': {k: [] for k in KINDS}, 'incons': {k: [] for k in KINDS},
         'incons_po': {k: [] for k in KINDS}, 'top1': {k: [] for k in KINDS}, 'auroc': []}
    for n, d in enumerate(test):
        ho, po = run(model, d)
        go = gates(d)
        mae_o = np.abs(po[go] - d['prob'][go]).mean()
        R['mae_orig'].append(mae_o)
        zo = ho[go] / (np.linalg.norm(ho[go], axis=1, keepdims=True) + 1e-9)
        pos_o = set((d['pos'] >> 1).tolist())
        for k in KINDS:
            v = tvars[n][k]
            s = v['var']
            hv, pv = run(model, s)
            gv = gates(s)
            R['mae'][k].append(np.abs(pv[gv] - s['prob'][gv]).mean())
            I, J, P = v['I'], v['J'], v['P']
            if len(I) == 0:
                continue
            pj = np.where(P == 1, 1 - pv[J], pv[J])
            diff = np.abs(po[I] - pj)
            R['incons'][k].append(diff.mean())
            rs = shash(d)[I] != shash(s)[J]
            R['frac_rs'].append(rs.mean())
            if rs.any():
                R['incons_rs'].append(diff[rs].mean())
                R['err_rs'].append(np.abs(po[I[rs]] - d['prob'][I[rs]]).mean())
            m = np.array([i in pos_o for i in I])
            if m.any():
                R['incons_po'][k].append(diff[m].mean())
            # retrieval: same-polarity matched variant nodes -> nearest original gate by cosine
            keep = P == 0
            if keep.sum() > 0:
                zj = hv[J[keep]]; zj = zj / (np.linalg.norm(zj, axis=1, keepdims=True) + 1e-9)
                sims = zj @ zo.T
                pred = go[sims.argmax(1)]
                R['top1'][k].append((pred == I[keep]).mean())
                rsk = rs[keep]
                if rsk.any():
                    R['top1_rs'].append((pred[rsk] == I[keep][rsk]).mean())
                # AUROC: equivalent pairs vs random non-equivalent pairs
                if k == KINDS[0] or True:
                    rng = np.random.default_rng(n)
                    pos_s = (zj * zo[I[keep] - go[0]]).sum(1)
                    ridx = rng.integers(0, len(go), size=len(pos_s))
                    neg_mask = go[ridx] != I[keep]
                    neg_s = sims[np.arange(len(pos_s)), ridx][neg_mask]
                    if len(neg_s):
                        allv = np.concatenate([pos_s, neg_s]); lab = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
                        r = allv.argsort().argsort() + 1
                        au = (r[lab == 1].sum() - len(pos_s) * (len(pos_s) + 1) / 2) / (len(pos_s) * len(neg_s))
                        R['auroc'].append(au)
    S = {}
    S['mae_orig'] = float(np.mean(R['mae_orig']))
    for key in ('mae', 'incons', 'incons_po', 'top1'):
        S[key] = {k: float(np.mean(v)) for k, v in R[key].items() if len(v)}
        S[key + '_all'] = float(np.mean([x for v in R[key].values() for x in v]))
    S['auroc'] = float(np.mean(R['auroc']))
    S['incons_ratio'] = S['incons_all'] / S['mae_orig']
    for key in ('incons_rs', 'err_rs', 'top1_rs', 'frac_rs'):
        S[key] = float(np.mean(R[key]))
    S['incons_rs_ratio'] = S['incons_rs'] / S['err_rs']
    return S


def evaluate_adv(model, advs):
    worst, avg, base = [], [], []
    for n, d in enumerate(test):
        _, po = run(model, d); go = gates(d)
        b0 = np.abs(po[go] - d['prob'][go]).mean()
        errs = []
        for v in advs[n]:
            s = v['var']; _, pv = run(model, s); gv = gates(s)
            errs.append(np.abs(pv[gv] - s['prob'][gv]).mean())
        if errs:
            worst.append(max(errs)); avg.append(np.mean(errs)); base.append(b0)
    return dict(adv_base=float(np.mean(base)), adv_avg=float(np.mean(avg)), adv_worst=float(np.mean(worst)),
                adv_worst_ratio=float(np.mean(worst) / np.mean(base)))


if __name__ == '__main__':
    tags = sys.argv[1:]
    if os.environ.get('NOADV'):
        torch.set_num_threads(1)
        res = {}
        for tag in tags:
            m = AIGEncoder(d=64, rounds=2, proj=tag.split('_s')[0].endswith('_ph')); m.load_state_dict(torch.load(f'm_{tag}.pt')); m.eval()
            res[tag] = evaluate(m)
            print(tag, json.dumps(res[tag]), flush=True)
        json.dump(res, open('results_fixed.json', 'w'), indent=1)
        sys.exit(0)
    if not os.path.exists('adv.pkl'):
        t = time.time()
        advs = [adv_variants(d, n) for n, d in enumerate(test)]
        pickle.dump(advs, open('adv.pkl', 'wb'))
        print('adv variants', sum(len(a) for a in advs), f'{time.time()-t:.0f}s', flush=True)
    advs = pickle.load(open('adv.pkl', 'rb'))
    res = json.load(open('results.json')) if os.path.exists('results.json') else {}
    for tag in tags:
        m = AIGEncoder(d=64, rounds=2, proj=tag.split('_s')[0].endswith('_ph')); m.load_state_dict(torch.load(f'm_{tag}.pt')); m.eval()
        S = evaluate(m); S.update(evaluate_adv(m, advs))
        res[tag] = S
        print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in S.items() if not isinstance(v, dict)}), flush=True)
        json.dump(res, open('results.json', 'w'), indent=1)
