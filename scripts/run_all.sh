#!/usr/bin/env bash
# Runs every experiment in priority order on ONE machine (see docs/REPRODUCE.md). Run from anywhere. Safe to rerun: finished steps are
# skipped (markers in results/.done_*), unfinished ones resume. Stops at the first failure; read the message.
#   GPUS=0,1,2,3,4,5,6,7 PER_GPU=3 bash scripts/run_all.sh    (run inside tmux)
#   STOP_AFTER=E1a bash scripts/run_all.sh                     (stop after a given experiment)
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; PER_GPU=${PER_GPU:-3}; THREADS=${THREADS:-2}; WORKERS=${WORKERS:-$(nproc)}
Q="python -m tools.run_queue --gpus $GPUS --threads $THREADS"
mkdir -p results logs
die() { echo; echo "STOPPED at step '$1'. Fix the cause (see message above, jobs/FAILED.txt, <run>/job.log), then rerun bash scripts/run_all.sh."; exit 1; }
step() {  # step NAME COMMAND...
  local n=$1; shift
  if [ -f results/.done_$n ]; then echo "== $n: already done"; return 0; fi
  echo "== $n: $(date '+%F %T')"; "$@" 2>&1 | tee -a logs/$n.log; local rc=${PIPESTATUS[0]}
  [ $rc -eq 0 ] || die $n
  touch results/.done_$n
}
phase() {  # phase EXP PHASE PER_GPU
  python -m tools.plan $1 $2 && $Q --per-gpu $3 jobs/$1_$2.jsonl
}
snapshot() { python -m tools.aggregate $1 || true; bash tools/pack_results.sh after_$1 >/dev/null 2>&1 || true; echo "   snapshot packed (certaudit_results_after_$1_*.tar.gz)"; }
stop_check() { if [ "${STOP_AFTER:-}" = "$1" ]; then echo "STOP_AFTER=$1 reached"; exit 0; fi; }

python -m tools.env_check > env_report.txt 2>&1; cat env_report.txt; grep -q "ENVIRONMENT OK" env_report.txt || die env_check

# ---- priority 1: environment reproduces the paper; DeepGate2 under both PI protocols
step E0_run        phase E0 all $PER_GPU
step E0_check      bash -c "python -m tools.repro_check && grep -q 'ALL PASS' results/E0/repro.md"
snapshot E0; stop_check E0
step E3_grid       phase E3 grid 2
step E3_select     python -m tools.select E3
step E3_final      phase E3 final 2
snapshot E3; stop_check E3
# ---- priority 2: the paper's claims at two widths, their mechanisms, and the H2 positive control
step E1a_lr        phase E1a lr $PER_GPU
step E1a_lr_sel    python -m tools.select E1a lr
step E1a_grid      phase E1a grid $PER_GPU
step E1a_grid_sel  python -m tools.select E1a grid
step E1a_final     phase E1a final $PER_GPU
snapshot E1; stop_check E1a
step E8_grid       phase E8 grid $PER_GPU
step E8_select     python -m tools.select E8 grid
step E8_final      phase E8 final $PER_GPU
snapshot E8; stop_check E8
step H2_data       python -m tools.gen_extra_variants --family h2 --workers $WORKERS
step E9_final      phase E9 all $PER_GPU
snapshot E9; stop_check E9
step E4_data       python -m tools.gen_extra_variants --family extra --workers $WORKERS   # needed by E12 and E4
step E12_final     phase E12 all $PER_GPU
snapshot E12; stop_check E12
# E13: build the large-circuit test suite from the EPFL benchmarks (CPU only, cones killed after 20 min each),
# then evaluate every E1a final model on it
step E13_data      bash -c "[ -f data/large.pkl ] || EPFL_DIR=${EPFL_DIR:-third_party/epfl_benchmarks} python -m tools.gen_large --workers $(( WORKERS < 64 ? WORKERS : 64 )) --timeout 1200"
step E13_eval      python -m tools.eval_large --workers 8
snapshot E13; stop_check E13
# ---- priority 3: independent DeepGate2 checkpoints, more widths, data scale, architectures
step E10_dg2       phase E10 all 1
snapshot E10; stop_check E10
step E1b_lr        phase E1b lr $PER_GPU
step E1b_lr_sel    python -m tools.select E1b lr
step E1b_grid      phase E1b grid $PER_GPU
step E1b_grid_sel  python -m tools.select E1b grid
step E1b_final     phase E1b final $PER_GPU
snapshot E1; stop_check E1b
step E4_lr         phase E4 lr $PER_GPU
step E4_lr_sel     python -m tools.select E4 lr
step E4_grid       phase E4 grid $PER_GPU
step E4_grid_sel   python -m tools.select E4 grid
step E4_final      phase E4 final $PER_GPU
snapshot E4; stop_check E4
step E2_lr         phase E2 lr $PER_GPU
step E2_lr_sel     python -m tools.select E2 lr
step E2_grid       phase E2 grid $PER_GPU
step E2_grid_sel   python -m tools.select E2 grid
step E2_final      phase E2 final $PER_GPU
snapshot E2; stop_check E2
# ---- priority 4 (optional, CPU-heavy): SAT sweeping with the scaled models
if [ "${RUN_E5:-0}" = "1" ]; then step E5 phase E5 all 4; snapshot E5; fi
python -m tools.aggregate all; bash tools/pack_results.sh final
echo "ALL DONE $(date '+%F %T')"
