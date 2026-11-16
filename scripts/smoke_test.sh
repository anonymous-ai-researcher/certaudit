#!/usr/bin/env bash
# Smoke test on CPU or GPU: train a small EqAlign encoder for one epoch, audit it on validation and on the first
# EPFL test circuits, and print the audit metrics. It checks that the pipeline runs; it does not reproduce a number.
#   bash scripts/smoke_test.sh            (a few minutes on a laptop CPU)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
RUN=runs/smoke/gru_d32_r2/eqalign_ph_l0.2_lr0.002_s0
N=${CERT_SMOKE_N:-20}
python scripts/verify_data.py
echo "== training (1 epoch, width 32, EqAlign, lambda 0.2)"
python -m core.train --method eqalign_ph --lam 0.2 --arch gru --d 32 --rounds 2 --lr 0.002 --seed 0 --epochs 1 --variants 1 --out $RUN
echo "== auditing (validation, and the first $N EPFL test circuits with every rewrite family)"
CERT_SMOKE_N=$N python -m core.evaluate --run $RUN --suites val,epfl
python - "$RUN" <<'EOF'
import json, pickle, sys, numpy as np
run = sys.argv[1]
v = json.load(open(run + '/val.json'))
print('validation:', {k: round(x, 4) for k, x in v.items() if isinstance(x, float)})
R = pickle.load(open(run + '/raw_epfl.pkl', 'rb'))
fams = sorted({k for r in R for k in r['kinds']})
print(f'{len(R)} test circuits, families: {fams} + held-out H1, H2 + {len(R[0]["adv"])} adversarial variants each')
def mean(key, kinds):
    x = [r['kinds'][k][key] for r in R for k in kinds if key in r['kinds'][k]]
    return float(np.mean(x)) if x else float('nan')
syn = [k for k in fams if str(k).startswith('S')]; inj = [k for k in fams if 'inj' in str(k)]
print(f"MAE original            {np.mean([r['mae_orig'] for r in R]):.4f}")
for name, ks in (('synthesis', syn), ('injection', inj)):
    print(f"{name:10s} rho {mean('incons_rs', ks) / mean('err_rs', ks):.2f}   Top-1 restructured {100 * mean('top1_rs', ks):.1f}%   chance {100 * mean('chance', ks):.1f}%")
h2 = [r['heldout']['H2'] for r in R if r.get('heldout')]
print(f"held-out H2 rho {np.mean([x['incons_rs'] for x in h2 if 'incons_rs' in x]) / np.mean([x['err_rs'] for x in h2 if 'err_rs' in x]):.2f}")
print('SMOKE TEST OK')
EOF
