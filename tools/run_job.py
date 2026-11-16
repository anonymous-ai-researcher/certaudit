"""Execute one job (called by run_queue.py; can also be called by hand with the job's JSON line).
Idempotent: training is skipped when model.pt exists; only missing evaluation suites are run."""
import json, os, subprocess, sys

SUITE_FILE = {'val': 'val.json', 'epfl': 'raw_epfl.pkl', 'iscas': 'raw_iscas.pkl', 'epfl_h': 'raw_epfl_h.pkl',
              'hyp': 'raw_hyp.pkl', 'abc': 'raw_abc.pkl', 'abc_iscas': 'raw_abc_iscas.pkl'}


def sh(cmd, log):
    with open(log, 'a') as f:
        f.write('$ ' + ' '.join(cmd) + '\n'); f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f'failed ({r.returncode}): {" ".join(cmd)}  -> see {log}')


def trained(out):
    return os.path.exists(os.path.join(out, 'model.pt')) and os.path.exists(os.path.join(out, 'train.json'))


def missing_suites(out, suites):
    return [s for s in suites.split(',') if not os.path.exists(os.path.join(out, SUITE_FILE[s]))]


def run(j):
    if os.environ.get('CERT_SMOKE_N') and not j['out'].startswith('runs/smoke'):
        raise SystemExit('CERT_SMOKE_N is set: it is only allowed for the smoke test. unset CERT_SMOKE_N')
    out = j['out']; os.makedirs(out, exist_ok=True); log = os.path.join(out, 'job.log'); py = sys.executable
    if j['kind'] == 'train':
        if not trained(out):
            sh([py, '-m', 'core.train'] + j['args'], log)
        miss = missing_suites(out, j['suites'])
        if miss:
            sh([py, '-m', 'core.evaluate', '--run', out, '--suites', ','.join(miss)], log)
    elif j['kind'] == 'dg2':
        suites = j['args'][j['args'].index('--suites') + 1]
        if missing_suites(out, suites) or not os.path.exists(os.path.join(out, 'train.json')):
            sh([py, '-m', 'core.dg2'] + j['args'], log)
    elif j['kind'] == 'dg2eval':                     # E17: evaluate a finished DeepGate2 model on new suites only
        suites = j['args'][j['args'].index('--suites') + 1]
        if missing_suites(out, suites):
            sh([py, '-m', 'core.dg2', '--eval-only'] + j['args'], log)
    elif j['kind'] == 'retrain':
        if not os.path.exists(os.path.join(out, 'model.pth')):
            sh([py, '-m', 'tools.dg2_retrain'] + j['args'], log)
    elif j['kind'] == 'noise':
        if not os.path.exists(os.path.join(out, 'noise.json')):
            sh([py, '-m', 'tools.noise', '--out', out] + j['args'], log)
    elif j['kind'] == 'sweep':
        if not os.path.exists(os.path.join(out, 'sweep.json')):
            sh([py, '-m', 'core.sweep'] + j['args'], log)
    elif j['kind'] == 'export':
        if not os.path.exists(os.path.join(out, 'export.json')):
            sh([py, '-m', 'tools.export_emb'] + j['args'], log)
    else:
        raise SystemExit(f'unknown job kind {j["kind"]}')


def is_done(j):
    out = j['out']
    if j['kind'] == 'train':
        return trained(out) and not missing_suites(out, j['suites'])
    if j['kind'] == 'dg2':
        return os.path.exists(os.path.join(out, 'train.json')) and not missing_suites(out, j['args'][j['args'].index('--suites') + 1])
    if j['kind'] == 'dg2eval':
        return not missing_suites(out, j['args'][j['args'].index('--suites') + 1])
    if j['kind'] == 'noise':
        return os.path.exists(os.path.join(out, 'noise.json'))
    if j['kind'] == 'retrain':
        return os.path.exists(os.path.join(out, 'model.pth'))
    if j['kind'] == 'sweep':
        return os.path.exists(os.path.join(out, 'sweep.json'))
    if j['kind'] == 'export':
        return os.path.exists(os.path.join(out, 'export.json'))
    return False


if __name__ == '__main__':
    run(json.loads(sys.argv[1]))
