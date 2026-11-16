"""Does certification matter for MEASUREMENT? (Experiment A showed it barely matters for training.)

Re-scores every model's audit with ground truth taken from 4,096-pattern simulation alone (first
signature candidate, no SAT), through eval2's own evaluate(), and compares with the proven
ground truth stored in raw_<tag>.pkl. If the audit's numbers or its ranking of regimes move,
the oracle has to be exact. Writes measure_cert.json."""
import os, sys, json, copy
os.environ['NOADV'] = '1'
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, torch
import eval2
from model import build
from analysis import load, group_summary, SYN, INJ
from make_sig_pairs import sig_match
from data import rep_of
eval2.H = None

if not os.path.exists('test_sig4096.json'):
    sim = []
    for n, d in enumerate(eval2.test):
        row = {}
        for k, v in eval2.tvars[n].items():
            I, J, P = sig_match(d, rep_of(v['var']), 64)
            row[k] = dict(I=I.tolist(), J=J.tolist(), P=P.tolist())
        sim.append(row)
    json.dump(sim, open('test_sig4096.json', 'w'))
sim = json.load(open('test_sig4096.json'))
proven = eval2.tvars
simv = [{k: dict(v, I=np.array(sim[n][k]['I'], np.int64), J=np.array(sim[n][k]['J'], np.int64),
                 P=np.array(sim[n][k]['P'], np.int64)) for k, v in proven[n].items()} for n in range(len(proven))]
wrong = tot = 0
for n in range(len(proven)):
    for k, v in proven[n].items():
        pv = set(zip(v['J'].tolist(), v['I'].tolist(), v['P'].tolist()))
        sv = list(zip(sim[n][k]['J'], sim[n][k]['I'], sim[n][k]['P']))
        tot += len(sv); wrong += sum(t not in pv for t in sv)
print(f'test set: {tot} simulation-labelled pairs, {wrong} ({100 * wrong / tot:.1f}%) not proven', flush=True)

OUT = json.load(open('measure_cert.json')) if os.path.exists('measure_cert.json') else {'wrong_frac': wrong / tot}
for tag in sys.argv[1:]:
    if tag in OUT:
        continue
    m = build(tag); m.load_state_dict(torch.load(f'v_{tag}.pt')); m.eval()
    eval2.tvars = simv
    Rs = eval2.evaluate(m)
    eval2.tvars = proven
    Rp = load(tag)
    res = {}
    for name, R in (('proven', Rp), ('sim', Rs)):
        g = group_summary(R, kinds=SYN + INJ)[0]
        res[name] = dict(top1=100 * g['top1_rs'], rho_syn=group_summary(R, kinds=SYN)[0]['rho'],
                         auroc=g['auroc_hard'])
    OUT[tag] = res
    json.dump(OUT, open('measure_cert.json', 'w'), indent=1)
    print(f"{tag:28s} Top-1 proven {res['proven']['top1']:.1f} sim {res['sim']['top1']:.1f} | "
          f"rho_syn proven {res['proven']['rho_syn']:.2f} sim {res['sim']['rho_syn']:.2f}", flush=True)
