"""Held-out transformation families (never seen in training), SAT-certified and SAT-matched.
H1: don't-care-based resubstitution/rewriting/refactoring + ESOP balancing.
H2: XOR re-encoding identity  x&y == (x^y)^(x|y)  at rate 0.3."""
import pickle, time
from multiprocessing import Pool
from aiglib import to_aig, from_aig, synth, inject, certify
from data import rep_of, rep_sample, match

def job(args):
    n, d = args
    rep = rep_of(d); a = to_aig(rep); out = {}
    for name in ('H1', 'H2'):
        b = synth(a, ['rsdc', 'besop', 'rw6dc', 'rfdc']) if name == 'H1' else inject(rep, 0.3, 50_000 + n, kinds=('xor',))
        ok = certify(a, b); rb = from_aig(b); st = {}
        I, J, P = match(d, rb, st)
        out[name] = dict(var=rep_sample(rb, d['src']), I=I, J=J, P=P, cert=ok, st=st)
    return out

if __name__ == '__main__':
    t = time.time()
    D = pickle.load(open('data.pkl', 'rb'))
    import signal
    class TO(Exception): pass
    def h(*a): raise TO()
    signal.signal(signal.SIGALRM, h)
    H = []
    for n, d in enumerate(D['test']):
        signal.alarm(60)
        try:
            H.append(job((n, d)))
        except TO:
            print('timeout on', n, flush=True); H.append(None)
        signal.alarm(0)
        if n % 50 == 0:
            print('done', n, f'{time.time()-t:.0f}s', flush=True)
    pickle.dump(H, open('heldout.pkl', 'wb'))
    print('timeouts', sum(h is None for h in H))
    H = [h for h in H if h is not None] if False else H
    for k in ('H1', 'H2'):
        Hk = [h for h in H if h is not None]
        print(k, 'cert ok', sum(h[k]['cert'] for h in Hk), '/', len(Hk), 'pairs', sum(len(h[k]['I']) for h in Hk),
              'gates', sum(len(h[k]['var']['fanins']) for h in Hk))
    print(f'{time.time()-t:.0f}s')
