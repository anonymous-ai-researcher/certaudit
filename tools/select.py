"""Selection on VALIDATION metrics only (no test result is ever read here). Rules, fixed in advance:
  lr     per architecture: the lr with the lowest baseline validation MAE (diverged runs excluded)
  grid   per architecture and regime: the weight with the highest validation restructured Top-1 among the
         weights whose validation MAE is within 25% of the baseline's (the paper's rule); if none qualifies,
         the weight with the lowest validation MAE, flagged 'no weight within 25%'. A pick at the largest or
         smallest weight of its grid is flagged 'edge' (reported, never re-tuned).
  E3     the (lambda, epochs) of EqAlign fine-tuning with the highest validation Top-1 among those within 25% of
         the control's validation MAE at the same number of epochs
The script STOPS (exit code 1, nothing written) if any expected run is missing: finish or rerun the phase first.
Diverged runs are the only allowed exclusion and are listed.
  python -m tools.select E1a lr | E1a grid | E8 grid | E3"""
import json, os, sys
from tools.plan import ARCHS, LRS, GRIDS, REGIMES, ABL_GRIDS, E3_GRID, E15_REGIMES, E15_LRS, aname, rname

missing, diverged = [], []


def val(run):
    tj, vj = os.path.join(run, 'train.json'), os.path.join(run, 'val.json')
    if not (os.path.exists(tj) and os.path.exists(vj)):
        missing.append(run); return None
    if json.load(open(tj))['config'].get('diverged'):
        diverged.append(run); return None
    return json.load(open(vj))


def pick(cand, base_mae, grid):
    ok = {k: v for k, v in cand.items() if v['val_mae'] <= 1.25 * base_mae}
    if ok:
        k = max(ok, key=lambda x: ok[x]['val_top1_rs']); flag = None
    else:
        k = min(cand, key=lambda x: cand[x]['val_mae']); flag = 'no weight within 25%'
    if k in (min(grid), max(grid)) and len(grid) > 1:
        flag = (flag + '; ' if flag else '') + 'edge'
    return k, flag


def main(exp, phase):
    path = f'runs/{exp}/selection.json'
    S = json.load(open(path)) if os.path.exists(path) else {}
    if exp in ('E1a', 'E1b', 'E1c', 'E2', 'E4', 'E15'):
        lrs = E15_LRS if exp in ('E15', 'E1c') else LRS; regimes = E15_REGIMES if exp == 'E15' else REGIMES
        for a in ARCHS[exp]:
            an = aname(*a); A = S.setdefault(an, {})
            if phase == 'lr':
                cand = {lr: v for lr in lrs if (v := val(f'runs/{exp}/{an}/{rname("base", None, lr, 0)}'))}
                if cand:
                    best = min(cand, key=lambda k: cand[k]['val_mae'])
                    A['lr'] = best; A['lr_candidates'] = {f'{k:g}': v for k, v in cand.items()}
            elif phase == 'grid':
                lr = A['lr']; b = val(f'runs/{exp}/{an}/{rname("base", None, lr, 0)}')
                if b is None:
                    continue
                A['base_val'] = b; A['lam'] = {'base': None, 'aug': None}; A['lam_candidates'] = {}; A['flags'] = {}
                for m in regimes:
                    if m in ('base', 'aug'):
                        if m == 'aug':
                            val(f'runs/{exp}/{an}/{rname("aug", None, lr, 0)}')
                        continue
                    cand = {lam: v for lam in GRIDS[m] if (v := val(f'runs/{exp}/{an}/{rname(m, lam, lr, 0)}'))}
                    A['lam_candidates'][m] = {f'{k:g}': v for k, v in cand.items()}
                    if cand:
                        A['lam'][m], f = pick(cand, b['val_mae'], GRIDS[m])
                        if f:
                            A['flags'][m] = f
    elif exp == 'E8':
        S1 = json.load(open('runs/E1a/selection.json'))
        for a in ARCHS['E8']:
            an = aname(*a); lr = S1[an]['lr']; b = S1[an]['base_val']; A = S.setdefault(an, {'lam': {}, 'flags': {}, 'lam_candidates': {}})
            for m, g in ABL_GRIDS.items():
                cand = {lam: v for lam in g if (v := val(f'runs/E8/{an}/{rname(m, lam, lr, 0)}'))}
                A['lam_candidates'][m] = {f'{k:g}': v for k, v in cand.items()}
                if cand:
                    A['lam'][m], f = pick(cand, b['val_mae'], g)
                    if f:
                        A['flags'][m] = f
    elif exp == 'E3':
        ctrl = {ep: val(f'runs/E3/grid/dg2_control_e{ep}_s0') for ep in E3_GRID['epochs']}
        cand = {}
        for ep in E3_GRID['epochs']:
            for lam in E3_GRID['lam']:
                v = val(f'runs/E3/grid/dg2_eqalign_l{lam:g}_e{ep}_s0')
                if v is not None and ctrl[ep] is not None:
                    cand[(lam, ep)] = dict(v, within=v['val_mae'] <= 1.25 * ctrl[ep]['val_mae'])
        if cand and not missing:
            ok = {k: v for k, v in cand.items() if v['within']}
            pk = max(ok, key=lambda k: ok[k]['val_top1_rs']) if ok else min(cand, key=lambda k: cand[k]['val_mae'])
            S = dict(lam=pk[0], epochs=pk[1], flag=None if ok else 'no setting within 25%',
                     control_val={str(k): v for k, v in ctrl.items()}, candidates={f'l{k[0]:g}_e{k[1]}': v for k, v in cand.items()})
    else:
        sys.exit(f'unknown experiment {exp}')
    if diverged:
        print('DIVERGED (excluded from selection):'); print('\n'.join('  ' + m for m in diverged))
    if missing:
        print('MISSING runs, selection NOT written. Rerun the phase (run_queue resumes) and check jobs/FAILED.txt:')
        print('\n'.join('  ' + m for m in missing)); sys.exit(1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'; json.dump(S, open(tmp, 'w'), indent=1, default=str); os.replace(tmp, path)
    show = S if exp == 'E3' else {k: {x: v.get(x) for x in ('lr', 'lam', 'flags')} for k, v in S.items()}
    print(json.dumps(show, indent=1, default=str)); print('written', path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'grid')
