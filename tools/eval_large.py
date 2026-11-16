"""E13: evaluate every finished E1a final run on the large-circuit suite data/large.pkl (test only).
Writes <run>/raw_large.pkl (same record format as raw_iscas.pkl). Resumable; GPU if available.
  python -m tools.eval_large [--exp E1a] [--workers 4]"""
import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def one(run):
    import torch
    from core.evaluate import load_run, Infer, eval_suite, data, _dump
    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m, cfg = load_run(run, dev)
    S = data('large.pkl')
    _dump(eval_suite(Infer(m, dev), S['test'], S['tvars'], None, S['ho']), os.path.join(run, 'raw_large.pkl'))
    return run


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--exp', default='E1a'); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    runs = sorted(os.path.dirname(p) for p in glob.glob(f'runs/{a.exp}/*/*/raw_epfl.pkl'))   # final runs only
    cfg = lambda r: json.load(open(os.path.join(r, 'train.json')))['config']
    runs = [r for r in runs if os.path.exists(os.path.join(r, 'train.json')) and 'arch' in cfg(r) and not cfg(r).get('diverged')]
    todo = [r for r in runs if not os.path.exists(os.path.join(r, 'raw_large.pkl'))]
    print(f'{len(runs)} final runs, {len(todo)} to evaluate on large.pkl', flush=True)
    if not runs:
        sys.exit('no finished final runs: run E1a first')
    bad = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, r) for r in todo]):
            try:
                print('done', f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                bad += 1; print('FAILED', type(e).__name__, e, flush=True)
    sys.exit(1 if bad else 0)
