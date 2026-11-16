"""Released DeepGate2 (python-deepgate, commit 173db75) in the paper's evaluation, and fine-tuning it on our
training circuits with and without EqAlign (the paper's experiment C, ext_audit/dg2_finetune.py, unchanged
except for the device, the EqAlign weight, the learning rate and the number of epochs being arguments).

  python -m core.dg2 --mode released --out runs/dg2_released                 # evaluate the released checkpoint
  python -m core.dg2 --mode control  --seed 0 --epochs 5 --out runs/dg2_ft_control_s0
  python -m core.dg2 --mode eqalign  --lam 0.2 --seed 0 --epochs 5 --out runs/dg2_ft_eqalign_l0.2_s0

Retrieval uses the projection head when the checkpoint has one (EqAlign) and h_f otherwise, as in the paper.

PRIMARY-INPUT PROTOCOL. DeepGate2 initialises the structural state of every primary input with random orthogonal
vectors drawn from np.random on EVERY forward pass. Evaluated that way (the paper's audit, protocol 'random'), an
original circuit and its variant receive different random PI vectors, so part of the measured disagreement is
evaluation noise: the same circuit retrieved against itself scores only ~41% Top-1. Protocol 'shared' gives a
circuit and its variants the same PI vectors (PIs are index-aligned): np.random is seeded with a fixed PI seed
before each graph's draw. Training: --pi random (python-deepgate's own behaviour, the paper's fine-tuning) or
--pi shared (each original/variant pair in a batch shares one freshly drawn PI seed).
Every model is evaluated under both protocols:
  <out>/raw_*.pkl, val.json      shared, PI seed 0 (primary)
  <out>/eval_shared_pi1|pi2/     shared, PI seeds 1 and 2 (sensitivity to the particular vectors; EPFL only)
  <out>/eval_random/             random, as in the paper (EPFL only)
  <out>/raw_hf/                  EqAlign checkpoints: retrieval on raw h_f instead of the head (shared, EPFL)
  <out>/selfcheck.json           same circuit against itself under both protocols (60 test cones)"""
import argparse, json, os, sys, time, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DG_DIR = os.environ.get('DEEPGATE_DIR', 'third_party/python-deepgate')
sys.path.insert(0, DG_DIR)
try:
    import torch_scatter  # noqa: F401
except ImportError:                                  # python-deepgate imports it but never calls it
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shims'))
import deepgate                                      # noqa: E402
import deepgate.model as _dgm                       # noqa: E402
from deepgate.utils.utils import generate_orthogonal_vectors  # noqa: E402
from deepgate.parser_func import parse_pyg_mlpgate  # noqa: E402
from torch_geometric.data import Batch              # noqa: E402
from core.train import split, DATA_DIR              # noqa: E402
from core import evaluate as EV                     # noqa: E402

GI = {'PI': 0, 'AND': 1, 'NOT': 2}


def to_graph(s, po_not=False):
    """Our sample -> DeepGate2 graph. Row (v-1) is variable v for PIs and ANDs (verified in the paper's dg2_impl)."""
    I, A = s['n_pi'], len(s['fanins'])
    x = [[i, GI['PI']] for i in range(I)] + [[I + i, GI['AND']] for i in range(A)]
    e, hasnot = [], {}

    def notnode(idx):
        if idx not in hasnot:
            x.append([len(x), GI['NOT']]); hasnot[idx] = len(x) - 1; e.append([idx, hasnot[idx]])
        return hasnot[idx]
    for g, (a, b) in enumerate(s['fanins']):
        for lit in (int(a), int(b)):
            src = (lit >> 1) - 1
            e.append([notnode(src) if lit & 1 else src, I + g])
    for lit in (s['pos'] if po_not else []):         # inference: as the paper's .aag reader (NOT rows on complemented POs)
        if int(lit) & 1 and (int(lit) >> 1) > 0:
            notnode((int(lit) >> 1) - 1)
    x, e = np.array(x), np.array(e)
    G = parse_pyg_mlpgate(x, e, [], [], [0.0] * len(x), [], [])
    G.tt_pair_index = torch.zeros((2, 0), dtype=torch.long); G.rc_pair_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.full((len(x),), -1.0)
    y[:I + A] = torch.tensor(np.asarray(s['prob'][1:1 + I + A], dtype=np.float32))
    G.y_prob = y
    return G


def _hs_init(G, hs, no_dim):
    """python-deepgate's generate_hs_init, except that a graph carrying pi_seed draws its PI vectors from
    np.random seeded with it (identical draws for graphs with the same seed and number of PIs). Graphs without
    pi_seed behave exactly as in python-deepgate."""
    seeds = getattr(G, 'pi_seed', None)
    B = 1 if G.batch is None else G.batch.max().item() + 1
    for b in range(B):
        pi_mask = (G.forward_level == 0) if G.batch is None else (G.batch == b) & (G.forward_level == 0)
        pi_node = G.forward_index[pi_mask]
        if seeds is not None:
            np.random.seed(int(seeds[b]))
        pi_vec = np.array(generate_orthogonal_vectors(len(pi_node), no_dim), dtype=np.float32)
        hs[pi_node] = torch.tensor(pi_vec, dtype=torch.float)
    return hs


_dgm.generate_hs_init = _hs_init


def _index(b):
    b.PIs = b.forward_index[(b.forward_level == 0) & (b.backward_level != 0)]
    b.POs = b.backward_index[(b.backward_level == 0) & (b.forward_level != 0)]
    b.no_connect = b.forward_index[(b.forward_level == 0) & (b.backward_level == 0)]
    return b


class InferDG2:
    def __init__(self, model, proj, dev, pi='shared', pi_seed=0):
        self.m, self.proj, self.dev, self.pi, self.pi_seed = model, proj, dev, pi, pi_seed

    @torch.no_grad()
    def __call__(self, s):
        if len(s['fanins']) == 0:                    # a variant that synthesis reduced to wires (ISCAS has 9):
            n = 1 + s['n_pi']                          # no gates to score; DeepGate2 cannot build an edgeless graph
            return np.zeros((n, 128), dtype=np.float32), np.full(n, 0.5, dtype=np.float32)
        G0 = to_graph(s, po_not=True)
        if self.pi == 'shared':
            G0.pi_seed = torch.tensor([self.pi_seed])
        g = _index(Batch.from_data_list([G0])).to(self.dev)
        hs, hf = self.m(g)
        pr = self.m.pred_prob(hf).squeeze(-1)
        z = self.proj(hf) if self.proj is not None else hf
        n = 1 + s['n_pi'] + len(s['fanins']); k = n - 1
        emb = np.zeros((n, z.shape[1]), dtype=np.float32); prob = np.zeros(n, dtype=np.float32)
        emb[1:] = z[:k].float().cpu().numpy(); prob[1:] = pr[:k].float().cpu().numpy()
        return emb, prob


def selfcheck(model, proj, dev, n=60):
    """Same circuit against itself: Top-1 of every gate (pessimistic ties) and mean |p1 - p2|, per protocol."""
    D = EV.data('data.pkl'); out = {}
    for pi in ('shared', 'random'):
        inf = InferDG2(model, proj, dev, pi, 0); hits, dp = [], []
        for d in D['test'][:n]:
            z1, p1 = inf(d); z2, p2 = inf(d); go = EV.gates(d)
            a_ = z1[go] / (np.linalg.norm(z1[go], axis=1, keepdims=True) + 1e-9)
            b_ = z2[go] / (np.linalg.norm(z2[go], axis=1, keepdims=True) + 1e-9)
            S_ = b_ @ a_.T; t_ = np.diag(S_)
            hits.append(float(((S_ >= t_[:, None] - 1e-7).sum(1) == 1).mean())); dp.append(float(np.abs(p1[go] - p2[go]).mean()))
        out[pi] = dict(self_top1=float(np.mean(hits)), self_abs_dp=float(np.mean(dp)), n_cones=len(hits))
    return out


def main(a):
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    os.makedirs(a.out, exist_ok=True)
    model = deepgate.Model()
    if a.ckpt:                                        # e.g. a DeepGate2 retrained from scratch (tools/dg2_retrain.py)
        model.load(a.ckpt)
    else:
        model.load_pretrained()
    model = model.to(dev)
    proj = None; log = []; t0 = time.time()
    lam = 0.0 if a.mode in ('released', 'control') else a.lam
    if a.mode != 'released':
        D = pickle.load(open(os.path.join(DATA_DIR, 'data.pkl'), 'rb'))
        TR, _ = split(len(D['train']))
        train = [D['train'][i] for i in TR]; pairs = [D['pairs'][i] for i in TR]
        GO = [to_graph(s) for s in train]; GV = [to_graph(p['var']) for p in pairs]
        proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)).to(dev)   # built always, as in the paper
        params = list(model.parameters()) + (list(proj.parameters()) if lam > 0 else [])
        if lam == 0:
            proj = None
        prng = np.random.default_rng(777 + a.seed)
        opt = torch.optim.Adam(params, lr=a.lr)
        BS, n = 16, len(GO)
        for ep in range(a.epochs):
            model.train()
            order = np.random.permutation(n); tot = totc = 0.0; nb = 0
            for k in range(0, n, BS):
                idx = order[k:k + BS]
                if a.pi == 'shared':                       # one PI seed per original/variant pair
                    for i in idx:
                        sd_ = torch.tensor([int(prng.integers(2 ** 31))]); GO[i].pi_seed = sd_; GV[i].pi_seed = sd_
                graphs = [GO[i] for i in idx] + [GV[i] for i in idx]
                offs = np.cumsum([0] + [g.num_nodes for g in graphs])[:-1]
                b = _index(Batch.from_data_list(graphs)).to(dev)
                hs, hf = model(b)
                pr = model.pred_prob(hf).squeeze(-1); y = b.y_prob; ok = y >= 0
                loss = F.l1_loss(pr[ok], y[ok]); lc = torch.zeros((), device=dev)
                if lam > 0:
                    Ia, Jb = [], []
                    for q, i in enumerate(idx):
                        pr_ = pairs[i]; keep = pr_['P'] == 0
                        Ia.append(offs[q] + pr_['I'][keep] - 1); Jb.append(offs[len(idx) + q] + pr_['J'][keep] - 1)
                    Ia = torch.from_numpy(np.concatenate(Ia)); Jb = torch.from_numpy(np.concatenate(Jb))
                    if len(Ia) > 512:
                        sel = torch.randperm(len(Ia))[:512]; Ia, Jb = Ia[sel], Jb[sel]
                    Ia, Jb = Ia.to(dev), Jb.to(dev)
                    za = F.normalize(proj(hf[Ia]), dim=-1); zb = F.normalize(proj(hf[Jb]), dim=-1)
                    lg = za @ zb.t() / 0.1; tg = torch.arange(len(Ia), device=dev)
                    lc = 0.5 * (F.cross_entropy(lg, tg) + F.cross_entropy(lg.t(), tg))
                    loss = loss + lam * lc
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                tot += float(loss); totc += float(lc); nb += 1
            log.append(dict(epoch=ep, loss=tot / nb, lc=totc / nb, sec=time.time() - t0))
            print(f'{a.out} ep{ep} loss {tot / nb:.4f} lc {totc / nb:.3f} {time.time() - t0:.0f}s', flush=True)
        torch.save(dict(model=model.state_dict(), proj=proj.state_dict() if proj is not None else None),
                   os.path.join(a.out, 'model.pt.tmp'))
        os.replace(os.path.join(a.out, 'model.pt.tmp'), os.path.join(a.out, 'model.pt'))
    model.eval()
    if proj is not None:
        proj.eval()
    cfg = dict(vars(a)); cfg.update(arch='deepgate2', method=f'dg2_{a.mode}', lam=lam, train_sec=time.time() - t0)
    if a.mode == 'released':
        cfg['pi'] = None
    EV.evaluate_run(a.out, InferDG2(model, proj, dev, 'shared', 0), a.suites.split(','))
    if 'epfl' in a.suites:
        for sub, inf in (('eval_shared_pi1', InferDG2(model, proj, dev, 'shared', 1)),
                         ('eval_shared_pi2', InferDG2(model, proj, dev, 'shared', 2)),
                         ('eval_random', InferDG2(model, proj, dev, 'random')),
                         ('raw_hf', InferDG2(model, None, dev, 'shared', 0) if proj is not None else None)):
            if inf is not None:
                os.makedirs(os.path.join(a.out, sub), exist_ok=True)
                np.random.seed(12345)
                EV.evaluate_run(os.path.join(a.out, sub), inf, ['epfl'])
    json.dump(selfcheck(model, proj, dev), open(os.path.join(a.out, 'selfcheck.json'), 'w'), indent=1)
    # train.json last: a run is complete only when it exists
    json.dump(dict(config=cfg, log=log), open(os.path.join(a.out, 'train.json.tmp'), 'w'), indent=1)
    os.replace(os.path.join(a.out, 'train.json.tmp'), os.path.join(a.out, 'train.json'))
    print('done', a.out, flush=True)


def eval_only(a):
    dev = torch.device(a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    model = deepgate.Model(); proj = None
    if a.mode == 'released':
        model.load(a.ckpt) if a.ckpt else model.load_pretrained()
    else:
        ck = torch.load(os.path.join(a.out, 'model.pt'), map_location='cpu')
        model.load_state_dict(ck['model'])
        if ck.get('proj') is not None:
            proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)); proj.load_state_dict(ck['proj'])
            proj = proj.to(dev).eval()
    model = model.to(dev).eval()
    EV.evaluate_run(a.out, InferDG2(model, proj, dev, 'shared', 0), a.suites.split(','))
    print('evaluated', a.out, a.suites, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['released', 'control', 'eqalign'], required=True)
    p.add_argument('--lam', type=float, default=0.2); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=5); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto'); p.add_argument('--suites', default='val,epfl,iscas')
    p.add_argument('--pi', choices=['random', 'shared'], default='shared', help='PI protocol during fine-tuning')
    p.add_argument('--ckpt', default=None, help='released mode: audit this checkpoint instead of the released one')
    p.add_argument('--out', required=True)
    p.add_argument('--eval-only', action='store_true',
                   help='E17: load the finished model of --out (model.pt; released: the released or --ckpt weights) and '
                        'evaluate --suites only (shared PI protocol, seed 0). Nothing is trained and train.json is not touched.')
    a = p.parse_args()
    eval_only(a) if a.eval_only else main(a)
