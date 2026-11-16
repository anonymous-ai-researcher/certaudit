"""E10: retrain DeepGate2 FROM SCRATCH with its official code and official training data (python-deepgate,
commit 173db75, examples/train.py: 60 epochs with weights [3, 1, 0], then 60 epochs with [3, 1, 2], lr 1e-4,
lr step 50), one seed per process on one GPU. The result is "DeepGate2 trained by us with the official code and
data", not the released checkpoint. Then audit it exactly like the released one (core/dg2.py --ckpt).

  python -m tools.dg2_retrain --seed 0 --out runs/E10/dg2_scratch_s0 [--epochs 60]
Data: third_party/dg2_train/train/{graphs.npz,labels.npz} from
  https://github.com/Ironprop-Stone/python-deepgate/releases/download/dataset/train.zip (MD5 97db0794c266202e87daa9ceb31ca864)"""
import argparse, os, random, shutil, sys, json, time
DG_DIR = os.environ.get('DEEPGATE_DIR', 'third_party/python-deepgate')
sys.path.insert(0, DG_DIR)
try:
    import torch_scatter  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core', 'shims'))
import numpy as np, torch
import deepgate
# python-deepgate reloads the dataset cache it has just written itself (dg2_parsed/) with torch.load; torch >= 2.6
# defaults to weights_only=True, which refuses its pickled PyG data class. The cache is created locally by this
# script from the official npz files, so it is trusted; only this process is affected.
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **dict(k, weights_only=k.get('weights_only', False)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--data', default='third_party/dg2_train/train')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: use only the first N circuits')
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    proc = os.path.join(a.out, 'dg2_parsed')            # python-deepgate caches its parsed dataset here
    dataset = deepgate.NpzParser(proc, os.path.join(a.data, 'graphs.npz'), os.path.join(a.data, 'labels.npz'))
    train_ds, val_ds = dataset.get_dataset()
    if a.limit:
        train_ds, val_ds = train_ds[:a.limit], val_ds[:max(2, a.limit // 4)]
    model = deepgate.Model()
    tr = deepgate.Trainer(model, training_id='train', save_dir=a.out, device=a.device, distributed=False)
    tr.set_training_args(prob_rc_func_weight=[3.0, 1.0, 0.0], lr=1e-4, lr_step=50)
    tr.train(a.epochs, train_ds, val_ds)
    tr.set_training_args(prob_rc_func_weight=[3.0, 1.0, 2.0], lr=1e-4, lr_step=50)
    tr.train(a.epochs, train_ds, val_ds)
    src = os.path.join(a.out, 'train', 'model_last.pth')
    shutil.copy(src, os.path.join(a.out, 'model.pth'))
    shutil.rmtree(proc, ignore_errors=True)
    json.dump(dict(seed=a.seed, epochs=a.epochs, n_train=len(train_ds), sec=time.time() - t0), open(os.path.join(a.out, 'retrain.json'), 'w'))
    print('saved', os.path.join(a.out, 'model.pth'), flush=True)
