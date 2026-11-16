"""E17 reproducibility check: rebuild the ABC families on this machine (data/abc_regen.pkl) and compare them with the
shipped data/abc.pkl (built on the authors' machine with the same pinned ABC commit). Writes data/abc_regen.stats.json.
The experiments always use the shipped data/abc.pkl; a mismatch is reported, never silently fixed."""
import argparse, json, os, pickle, subprocess, sys, numpy as np

DATA_DIR = os.environ.get('CERT_DATA', 'data')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=os.cpu_count()); a = ap.parse_args()
    if not os.path.exists(os.path.join(DATA_DIR, 'abc_regen.pkl')):
        r = subprocess.run([sys.executable, '-m', 'tools.gen_abc_variants', '--workers', str(a.workers), '--name', 'abc_regen'])
        if r.returncode:
            sys.exit('regeneration failed')
    A = pickle.load(open(os.path.join(DATA_DIR, 'abc.pkl'), 'rb')); B = pickle.load(open(os.path.join(DATA_DIR, 'abc_regen.pkl'), 'rb'))
    diff, tot = [], 0
    for su in ('epfl', 'iscas'):
        for n, (ra, rb) in enumerate(zip(A[su], B[su])):
            for f in sorted(set(ra) | set(rb)):
                tot += 1
                x, y = ra.get(f), rb.get(f)
                same = x is not None and y is not None and all(np.array_equal(x['var'][k], y['var'][k]) for k in ('fanins', 'pos')) \
                    and all(np.array_equal(x[k], y[k]) for k in ('I', 'J', 'P'))
                if not same:
                    diff.append(f'{su}/{n}/{f}')
    out = dict(variants_compared=tot, identical=tot - len(diff), different=len(diff), first_differences=diff[:20],
               abc_shipped=A.get('abc_version'), abc_here=B.get('abc_version'))
    json.dump(out, open(os.path.join(DATA_DIR, 'abc_regen.stats.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
    print('ABC data reproduced exactly' if not diff else f'WARNING: {len(diff)} variants differ (reported; the shipped data/abc.pkl is used)')
