"""E17: certified ABC rewrites of every TEST cone (EPFL and ISCAS), with SAT-proven gate correspondences, built exactly
like the paper's test variants (core/certdata.py: same certification, same matching) but rewritten by Berkeley ABC
instead of mockturtle. Nothing here is used for training or selection; it is a test-only family. CPU only, resumable.

  python -m tools.gen_abc_variants --workers 64            # writes data/abc.pkl (+ abc.stats.json)
  python -m tools.gen_abc_variants --limit 10 --workers 4  # smoke: data/abc_smoke.pkl

Families (fixed before any model is evaluated on them; scripts spelled out so that no abc.rc alias is needed):
  A1  resyn2        ABC's standard area-recovery script (abc.rc alias 'resyn2')
  A2  dc2           ABC's built-in combinational synthesis command
  A3  compress2rs   ABC's alias 'compress2rs' (balance / resubstitution / rewriting / refactoring, level-preserving)
  A4  lutmap        6-input LUT mapping with SAT-based don't-care optimisation (mfs2), then strashed back to an AIG:
                    the round trip of an FPGA flow, which restructures far more than the other three
Each variant must pass TWO independent equivalence checks to be kept: mockturtle's SAT-based equivalence checking
(the paper's certification, core/aiglib.certify) AND ABC's own 'cec'. Gate correspondences are then proven one by one
by SAT (core/certdata.match), exactly as for the paper's families. A failed check drops the variant and is counted."""
import argparse, json, os, pickle, re, shutil, subprocess, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import numpy as np

DATA_DIR = os.environ.get('CERT_DATA', 'data')
ABC = os.environ.get('ABC_BIN', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'abc', 'abc'))
SCRIPTS = {
    'A1_resyn2': 'balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance',
    'A2_dc2': 'dc2',
    'A3_compress2rs': ('balance -l; resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; balance -l; '
                       'resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; balance -l; '
                       'resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l; balance -l'),
    'A4_lutmap': 'if -K 6; mfs2; strash',
}
FAMS = list(SCRIPTS)
D = S = None


def _init():
    global D, S
    D = pickle.load(open(os.path.join(DATA_DIR, 'data.pkl'), 'rb'))
    S = pickle.load(open(os.path.join(DATA_DIR, 'iscas.pkl'), 'rb'))


def abc(cmd, timeout=120):
    r = subprocess.run([ABC, '-q', cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def abc_rewrite(src_aig, dst_aig, script):
    code, out = abc(f'read_aiger {src_aig}; strash; {script}; strash; write_aiger {dst_aig}')
    if code != 0 or not os.path.exists(dst_aig):
        raise RuntimeError('abc failed: ' + out[-300:])


def abc_cec(a, b):
    code, out = abc(f'cec {a} {b}')
    if re.search(r'Networks are equivalent', out):
        return True
    if re.search(r'Networks are NOT EQUIVALENT|are not equivalent|NOT EQUIVALENT', out, re.I):
        return False
    raise RuntimeError('abc cec gave no verdict: ' + out[-300:])


def job(suite, n):
    import signal
    from aigverse.io import write_aiger, read_aiger_into_aig
    from core.certdata import match, rep_sample, rep_of
    from core.aiglib import to_aig, from_aig, certify

    def _to(*_):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _to); signal.alarm(900)
    d = (D['test'] if suite == 'epfl' else S['test'])[n]
    row, st = {}, dict(cert_fail=0, cec_fail=0, abc_error=0, identical=0)
    tmp = tempfile.mkdtemp(prefix='abc_')
    try:
        a = to_aig(rep_of(d)); fa = os.path.join(tmp, 'a.aig'); write_aiger(a, fa)
        for fam, script in SCRIPTS.items():
            fb = os.path.join(tmp, f'{fam}.aig')
            try:
                abc_rewrite(fa, fb, script)
                b = read_aiger_into_aig(fb)
            except Exception:  # noqa: BLE001
                st['abc_error'] += 1; continue
            rb = from_aig(b)
            if rb.n_pi != d['n_pi'] or len(rb.pos) != len(d['pos']):
                st['abc_error'] += 1; continue
            ok_mt = bool(certify(a, b)); ok_abc = abc_cec(fa, fb)
            if not ok_mt:
                st['cert_fail'] += 1
            if not ok_abc:
                st['cec_fail'] += 1
            if not (ok_mt and ok_abc):
                continue
            s = rep_sample(rb, d['src'])
            if s['n_pi'] == d['n_pi'] and np.array_equal(s['fanins'], d['fanins']) and np.array_equal(s['pos'], d['pos']):
                st['identical'] += 1                     # kept (a legitimate outcome) but counted
            I, J, P = match(d, rb)
            row[fam] = dict(var=s, I=I, J=J, P=P, cert=True, cec=True, n_gates=len(s['fanins']), n_gates_orig=len(d['fanins']))
    finally:
        signal.alarm(0); shutil.rmtree(tmp, ignore_errors=True)
    return suite, n, row, st


def abc_version():
    try:
        _, out = abc('version')
        c = subprocess.run(['git', '-C', os.path.dirname(ABC), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        return out.strip().splitlines()[-1] + (f' @ {c}' if c else '')
    except Exception as e:  # noqa: BLE001
        return f'unknown ({e})'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--name', default=None, help='output name in data/ (default abc, or abc_smoke with --limit)')
    a = ap.parse_args()
    if not os.path.exists(ABC):
        raise SystemExit(f'ABC binary not found at {ABC} (set ABC_BIN)')
    _init()
    name = a.name or ('abc' + ('_smoke' if a.limit else ''))
    NE = a.limit or len(D['test']); NI = a.limit or len(S['test'])
    tasks = [('epfl', n) for n in range(NE)] + [('iscas', n) for n in range(NI)]
    part = os.path.join(DATA_DIR, name + '.partial.pkl')
    done = pickle.load(open(part, 'rb')) if os.path.exists(part) else {}
    tries = {}; t0 = time.time()
    while True:
        todo = [t for t in tasks if t not in done]
        if not todo:
            break
        try:
            with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
                futs = {ex.submit(job, *t): t for t in todo}
                for i, f in enumerate(as_completed(futs)):
                    t = futs[f]
                    try:
                        su, n, row, st = f.result(); done[t] = (row, st)
                    except BrokenProcessPool:
                        raise
                    except Exception as e:  # noqa: BLE001
                        tries[t] = tries.get(t, 0) + 1
                        if tries[t] >= 2:
                            done[t] = ({}, dict(error=type(e).__name__))
                    if i % 100 == 0:
                        pickle.dump(done, open(part, 'wb'))
                        print(f'{len(done)}/{len(tasks)} cones, {time.time() - t0:.0f}s', flush=True)
        except BrokenProcessPool:
            for t in todo:
                if t not in done:
                    tries[t] = tries.get(t, 0) + 1
                    if tries[t] >= 3:
                        done[t] = ({}, dict(error='crash'))
            print('worker crashed; restarting pool', flush=True)
        pickle.dump(done, open(part, 'wb'))
    out = dict(epfl=[done[('epfl', n)][0] for n in range(NE)], iscas=[done[('iscas', n)][0] for n in range(NI)],
               scripts=SCRIPTS, abc_version=abc_version())
    pickle.dump(out, open(os.path.join(DATA_DIR, name + '.pkl'), 'wb'))
    stats = [v[1] for v in done.values()]
    summ = dict(file=name + '.pkl', abc=out['abc_version'], cones=len(tasks), seconds=round(time.time() - t0))
    for su in ('epfl', 'iscas'):
        for fam in FAMS:
            V = [r[fam] for r in out[su] if fam in r]
            summ[f'{su}/{fam}'] = dict(
                variants=len(V), proven_pairs=int(sum(len(v['I']) for v in V)),
                gates_ratio=round(float(np.mean([v['n_gates'] / max(v['n_gates_orig'], 1) for v in V])), 3) if V else None,
                frac_matched=round(float(np.mean([len(np.unique(v['J'])) / max(v['n_gates'], 1) for v in V])), 3) if V else None)
    for k in ('cert_fail', 'cec_fail', 'abc_error', 'identical'):
        summ[k] = int(sum(s.get(k, 0) for s in stats))
    summ['cones_failed'] = int(sum('error' in s for s in stats))
    json.dump(summ, open(os.path.join(DATA_DIR, name + '.stats.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
