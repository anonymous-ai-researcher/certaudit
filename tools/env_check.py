"""Environment check: packages, GPUs, data integrity (MD5), DeepGate2 checkpoint, SAT and synthesis toolchain.
  python -m tools.env_check        (prints PASS/FAIL per item; send the output back if anything fails)"""
import hashlib, json, os, platform, sys

ok_all = True


def check(name, fn):
    global ok_all
    try:
        msg = fn(); print(f'[PASS] {name}: {msg}')
    except Exception as e:  # noqa: BLE001
        ok_all = False; print(f'[FAIL] {name}: {type(e).__name__}: {e}')


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pkgs():
    import numpy, scipy, torch
    v = dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__, torch=torch.__version__)
    assert int(numpy.__version__.split('.')[0]) >= 2, 'numpy >= 2 is required to read the data pickles'
    return v


def gpus():
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        return 'no CUDA GPU visible (fine only for the CPU smoke test)'
    return f'{n} x {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}, cpu cores {os.cpu_count()}'


def data():
    man = json.load(open('configs/data_manifest.json')); D = os.environ.get('CERT_DATA', 'data'); bad = []
    for f, h in man.items():
        p = os.path.join(D, f)
        if not os.path.exists(p):
            bad.append(f + ' missing')
        elif md5(p) != h:
            bad.append(f + ' md5 mismatch')
    opt = 'configs/data_manifest_optional.json'      # add-on files (E13 large.pkl): checked only when present
    n_opt = 0
    if os.path.exists(opt):
        for f, h in json.load(open(opt)).items():
            p = os.path.join(D, f)
            if os.path.exists(p):
                n_opt += 1
                if md5(p) != h:
                    bad.append(f + ' md5 mismatch (re-download this add-on file)')
    assert not bad, '; '.join(bad)
    return f'{len(man)} files verified' + (f', {n_opt} add-on file(s) verified' if n_opt else '')


def toolchain():
    import pickle
    from core.certdata import variant, match
    d = pickle.load(open(os.path.join(os.environ.get('CERT_DATA', 'data'), 'data.pkl'), 'rb'))['train'][0]
    rb, ok = variant(d, 'S1', seed=0); I, J, P = match(d, rb)
    assert ok, 'certification failed'
    return f'aigverse synthesis + CaDiCaL certification OK ({len(I)} proven pairs on a test cone)'


def deepgate():
    dg = os.environ.get('DEEPGATE_DIR', 'third_party/python-deepgate')
    p = os.path.join(dg, 'deepgate', 'pretrained', 'model.pth')
    assert os.path.exists(p), f'{p} not found (see README step 3)'
    h = md5(p); assert h == '09c214ae3fa7c654451f62d4ecb0c6e2', f'model.pth md5 {h} differs from the paper\'s'
    import core.dg2  # noqa: F401  (imports python-deepgate and torch_geometric)
    return 'python-deepgate importable, released checkpoint md5 matches'


def dg2data():
    p = os.path.join('third_party', 'dg2_train', 'train.zip')
    assert os.path.exists(p), f'{p} not found (README step 3; needed for E10 only)'
    assert md5(p) == '97db0794c266202e87daa9ceb31ca864', 'train.zip md5 differs'
    assert os.path.exists(os.path.join('third_party', 'dg2_train', 'train', 'graphs.npz')), 'unzip train.zip there'
    return 'official DeepGate2 training data present'


def ram():
    kb = int(next(l for l in open('/proc/meminfo') if l.startswith('MemTotal')).split()[1])
    return f'{kb / 1e6:.0f} GB RAM, {os.cpu_count()} CPU cores (send these numbers back)'


check('packages', pkgs); check('GPUs', gpus); check('host', ram); check('data', data); check('SAT + synthesis', toolchain)
check('DeepGate2', deepgate); check('DeepGate2 training data', dg2data)
print('\nENVIRONMENT OK' if ok_all else '\nSOME CHECKS FAILED (fix them before running; E3 needs only the DeepGate2 item)')
sys.exit(0 if ok_all else 1)
