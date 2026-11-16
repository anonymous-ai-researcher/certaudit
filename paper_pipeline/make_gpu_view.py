"""Build work_gpu: the paper's analysis scripts run unchanged on the GPU (8-GPU machine) results.

Every main-text model result (Table I, Figs. 2 and 3, RQ1/RQ2 numbers, ISCAS, the h-space retrieval, the frozen-encoder
probes, the no-projection and DeepGate2-faithful TT-sim ablations, the noise floor and the gap closed) comes from the
E1a and E8 final runs at width 64 (5 seeds), mapped onto the file names the scripts read. Everything else is a symlink
to this directory: model-free data (variants, proofs, signature statistics), and the results that exist only from the
CPU development environment until their GPU counterparts arrive (SAT sweep: E5; bidirectional encoder: E2; signature
pairs, Table II test columns, recall@k, Fig. 1 example: E14; confirmatory design hyp: kept from the CPU run, where it
was evaluated once after every other result was fixed).

  python3 make_gpu_view.py <extracted runs/ directory>"""
import os, sys, json, glob, shutil
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__)); DST = SRC + '_gpu'
RUNS = sys.argv[1]
A = f'{RUNS}/E1a/gru_d64_r2'; E8 = f'{RUNS}/E8/gru_d64_r2'
SEL = {'ttsim_ph': '0.03', 'circ_ph': '0.2', 'eqalign_ph': '0.2', 'l1align': '0.0003', 'l1tt': '0.003'}   # E1a selection.json
MAP = {}                                       # destination name -> source file
for i in range(5):
    for dst, src in (('base', 'base_lr0.002'), ('aug', 'aug_lr0.002'), ('ttsim_ph_l0.03', 'ttsim_ph_l0.03_lr0.002'),
                     ('circ_ph_l0.2', 'circ_ph_l0.2_lr0.002'), ('eqalign_ph_l0.2', 'eqalign_ph_l0.2_lr0.002'),
                     ('l1align_l0.0003', 'l1align_l0.0003_lr0.002'), ('l1tt_l0.003', 'l1tt_l0.003_lr0.002')):
        MAP[f'raw_{dst}_s{i}.pkl'] = f'{A}/{src}_s{i}/raw_epfl.pkl'
        if dst in ('base', 'aug', 'ttsim_ph_l0.03', 'circ_ph_l0.2', 'eqalign_ph_l0.2'):
            MAP[f'rawis_{dst}_s{i}.pkl'] = f'{A}/{src}_s{i}/raw_iscas.pkl'
    MAP[f'hret_raw/raw_eqalign_ph_l0.2_s{i}_hret.pkl'] = f'{A}/eqalign_ph_l0.2_lr0.002_s{i}/raw_epfl_h.pkl'
    MAP[f'raw_ttsimf_ph_l0.03_s{i}.pkl'] = f'{E8}/ttsimf_ph_l0.03_lr0.002_s{i}/raw_epfl.pkl'
    MAP[f'raw_eqalign_l0.2_s{i}.pkl'] = f'{E8}/eqalign_l0.2_lr0.002_s{i}/raw_epfl.pkl'
    MAP[f'raw_eqalign_probebase_ph_l0.2_s{i}.pkl'] = f'{E8}/eqalign_ph_l0.2_lr0.002_probebase_s{i}/raw_epfl.pkl'
    MAP[f'raw_eqalign_probeaug_ph_l0.2_s{i}.pkl'] = f'{E8}/eqalign_ph_l0.2_lr0.002_probeaug_s{i}/raw_epfl.pkl'
# E2 (second machine): bidirectional encoder and transformer, the main width of each family, weights E2 selected
E2 = os.environ.get('E2RUNS')
if E2:
    for i in range(5):
        for tag, arch, lam in (('bi', 'bi_d64_r2', '0.1'), ('tf', 'tf_d128_r2', '0.1')):
            for dst, src in (('base', 'base_lr0.002'), ('aug', 'aug_lr0.002'), (f'eqalign_ph_l{lam}', f'eqalign_ph_l{lam}_lr0.002')):
                MAP[f'raw_{tag}_{dst}_s{i}.pkl'] = f'{E2}/E2/{arch}/{src}_s{i}/raw_epfl.pkl'
missing = [s for s in MAP.values() if not os.path.exists(s)]
assert not missing, missing[:5]

# model results of the CPU environment that the GPU view replaces: never symlinked (no silent mixing)
REPLACED = lambda n: (n.startswith(('raw_base_s', 'raw_aug_s', 'raw_ttsim_ph_', 'raw_circ_ph_', 'raw_eqalign_ph_l0.2_s',
                                    'raw_l1align_', 'raw_l1tt_', 'raw_ttsimf_', 'raw_eqalign_l', 'raw_eqalign_probe', 'rawis_'))
                      or (bool(os.environ.get('E2RUNS')) and n.startswith(('raw_bi_', 'raw_tf_')))
                      or (bool(os.environ.get('E14RUNS')) and (n.startswith(('raw_eqalign_ph_l', 'raw_eqalign_sig', 'sweep2_v_'))
                                                                or n in ('recall_at_k.json', 'fig1_data.json', 'fig1c_sims.json', 'fig1cand')))
                      or n in ('noise_floor.json', 'rho_null2.json', 'numbers.json', 'iscas_design_test.json'))
if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(DST)
for n in os.listdir(SRC):
    if n in ('paper',) or REPLACED(n) or n.startswith('.'):
        continue
    os.symlink(os.path.join(SRC, n), os.path.join(DST, n))
os.symlink(os.path.join(SRC, 'paper'), os.path.join(DST, 'paper'))   # outputs go to the paper
os.makedirs(f'{DST}/hret_raw_gpu')
# downloaded pickles are read only through the restricted unpickler (numpy arrays, scalars, dicts, lists) and
# re-serialized here, so the analysis scripts' plain pickle.load never touches a downloaded file
sys.path.insert(0, '${WORK}')
from safe_pickle import load as safe_load
import pickle
for dst, src in MAP.items():
    if dst.startswith('hret_raw/'):
        dst = dst.replace('hret_raw/', 'hret_raw_gpu/')
    pickle.dump(safe_load(src), open(os.path.join(DST, dst), 'wb'))
MAP = {k: os.path.join(DST, k.replace('hret_raw/', 'hret_raw/')) for k in MAP}
os.remove(f'{DST}/hret_raw'); os.rename(f'{DST}/hret_raw_gpu', f'{DST}/hret_raw')
# noise floor: tools/noise.py on the 5 baseline seeds
nz = json.load(open(f'{A}/noise/noise.json'))
json.dump({'base': nz['base'] if 'base' in nz else nz['noise_floor']}, open(f'{DST}/noise_floor.json', 'w'))
# gap closed from the evaluator's in-pipeline permutation (incons_perm): per seed, circuit means
sys.path.insert(0, SRC)
from analysis import group_metrics, SYN, INJ
import pickle
rn = {}
for dst in MAP:
    if not dst.startswith('raw_') or dst.startswith(('raw_ttsimf', 'raw_eqalign_l', 'raw_eqalign_probe', 'raw_bi_', 'raw_tf_')):
        continue
    R = pickle.load(open(MAP[dst], 'rb')); tag = dst[4:-4]; rn[tag] = {}
    for fam, K in (('syn', SYN), ('inj', INJ)):
        inc, perm, err = (np.array([np.nanmean([r['kinds'][str(k)].get(q, np.nan) for k in K if str(k) in r['kinds']])
                                    for r in R]) for q in ('incons_rs', 'incons_perm', 'err_rs'))
        rn[tag][fam] = dict(rho=float(np.nanmean(inc) / np.nanmean(err)), rho_perp=float(np.nanmean(perm) / np.nanmean(err)),
                            gap=float(1 - np.nanmean(inc) / np.nanmean(perm)))
json.dump(rn, open(f'{DST}/rho_null2.json', 'w'), indent=1)
# validation results of every E1a grid run (seed 0) for the selection table, keyed like the CPU run's file
VR = {}
for vj in sorted(glob.glob(f'{A}/*_s0/val.json')):
    run = os.path.basename(os.path.dirname(vj)); tag = run.replace('_lr0.002', '')
    if '_lr' in tag:
        continue                                         # learning-rate candidates (baseline only) are listed elsewhere
    VR[tag] = json.load(open(vj))
for vj in sorted(glob.glob(f'{E8}/eqalign_l*_s0/val.json')):     # EqAlign without projection head (E8)
    VR[os.path.basename(os.path.dirname(vj)).replace('_lr0.002', '')] = json.load(open(vj))
if os.path.islink(f'{DST}/val_results.json'):
    os.remove(f'{DST}/val_results.json')
json.dump(VR, open(f'{DST}/val_results.json', 'w'), indent=1)
# ---- E14 and E5 (second machine, on copies of the E1a final and grid models of the first)
E14 = os.environ.get('E14RUNS')
if E14:
    A2 = f'{E14}/E1a/gru_d64_r2'; n14 = 0
    # Table II: test results of the weight-grid runs (seed 0); the final runs' seed 0 is already mapped
    for d in sorted(glob.glob(f'{A2}/*_s0')):
        tag = os.path.basename(d).replace('_lr0.002', ''); dst = f'{DST}/raw_{tag}.pkl'
        if os.path.exists(f'{d}/raw_epfl.pkl') and not os.path.exists(dst) and '_lr' not in tag:
            pickle.dump(safe_load(f'{d}/raw_epfl.pkl'), open(dst, 'wb')); n14 += 1
    # simulation-signature positives instead of proofs (E14a), 5 seeds
    for bits in (64, 4096):
        for f in sorted(glob.glob(f'{E14}/E14/gru_d64_r2/eqalign_ph_l0.2_lr0.002_sig{bits}_s*/raw_epfl.pkl')):
            sd = f.split('_s')[-1].split('/')[0]
            pickle.dump(safe_load(f), open(f'{DST}/raw_eqalign_sig{bits}_ph_l0.2_s{sd}.pkl', 'wb')); n14 += 1
    # SAT sweeping (E5): same record format as the CPU run's files
    for tag, run in (('base', 'base_lr0.002'), ('aug', 'aug_lr0.002'), ('ttsim_ph_l0.03', 'ttsim_ph_l0.03_lr0.002'),
                     ('circ_ph_l0.2', 'circ_ph_l0.2_lr0.002'), ('eqalign_ph_l0.2', 'eqalign_ph_l0.2_lr0.002')):
        for f in sorted(glob.glob(f'{A2}/{run}_s*/sweep.json')):
            sd = f.split('_s')[-1].split('/')[0]
            json.dump(json.load(open(f)), open(f'{DST}/sweep2_v_{tag}_s{sd}.json', 'w')); n14 += 1
    for f in glob.glob(f'{SRC}/sweep2_v_*.json'):            # the CPU run's sweeps, for its timings (Fig. 4b)
        shutil.copy(f, f'{DST}/cpu_' + os.path.basename(f))
    # recall@k (E14c), 5 seeds
    RK = {}
    for m, run in (('base', 'base_lr0.002'), ('aug', 'aug_lr0.002'), ('tt', 'ttsim_ph_l0.03_lr0.002'),
                   ('circ', 'circ_ph_l0.2_lr0.002'), ('ours', 'eqalign_ph_l0.2_lr0.002')):
        R_ = [json.load(open(f)) for f in sorted(glob.glob(f'{A2}/{run}_s*/recall.json'))]
        X_ = np.array([r['recall'] for r in R_])
        RK[m] = dict(mean=X_.mean(0).tolist(), sd=X_.std(0, ddof=1).tolist(), k=R_[0]['k'], n_seeds=len(R_))
    json.dump(RK, open(f'{DST}/recall_at_k.json', 'w'), indent=1)
    # Fig. 1: the same example pair, seed-0 models (as the CPU run's figure)
    F = {m: json.load(open(f'{A2}/{run}_s0/fig1.json')) for m, run in (('base', 'base_lr0.002'), ('ours', 'eqalign_ph_l0.2_lr0.002'))}
    old = json.load(open(f'{SRC}/fig1_data.json'))['example']
    assert all(F[m]['circuit'] == old['circuit'] and F[m]['v'] == old['v'] and F[m]['u'] == old['u'] for m in F)
    nb = lambda f: dict(curve=f['curve'], rank=f['rank'], n_cand=f['n_cand'], sim_true=f['sim_true'], sim_top=f['sim_top'],
                        top_is_true=f['top_is_true'], pts=f['pts'], u=f['u_pt'], idx_true=f['idx_true'])
    ex = dict(old, base=[F['base']['p_orig'], F['base']['p_var']], ours=[F['ours']['p_orig'], F['ours']['p_var']])
    json.dump(dict(example=ex, base=nb(F['base']), ours=nb(F['ours'])), open(f'{DST}/fig1_data.json', 'w'), indent=1)
    # panel C: sorted cosines with v at its rank (the panel uses the set of cosines and v's value only)
    json.dump({m: dict(sims=F[m]['curve'], idx_true=F[m]['rank'] - 1, rank=F[m]['rank'], pca=F[m]['pts'] + [F[m]['u_pt']])
               for m in F}, open(f'{DST}/fig1c_sims.json', 'w'))
    shutil.copytree(f'{SRC}/fig1cand', f'{DST}/fig1cand', symlinks=False,
                    ignore=shutil.ignore_patterns('*.aux', '*.log', 'cv_*'))
    # hyp, GPU models (supplementary replication; the confirmatory run stays the CPU one)
    HG = {}
    for tag, run in (('Base', 'base_lr0.002'), ('Aug', 'aug_lr0.002'), ('TT', 'ttsim_ph_l0.03_lr0.002'),
                     ('Circ', 'circ_ph_l0.2_lr0.002'), ('Ours', 'eqalign_ph_l0.2_lr0.002')):
        HG[tag] = []
        for f in sorted(glob.glob(f'{A2}/{run}_s*/raw_hyp.pkl')):
            HG[tag].append(safe_load(f))
    pickle.dump(HG, open(f'{DST}/hyp_gpu.pkl', 'wb'))
    print(f'E14/E5: {n14} result files, recall {list(RK)}, fig1 ranks {F["base"]["rank"]}/{F["ours"]["rank"]}, hyp seeds {[len(v) for v in HG.values()]}')
print(f'{DST}: {len(MAP)} GPU result files mapped; {len(VR)} validation runs; noise floor {nz}')
