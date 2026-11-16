#!/usr/bin/env bash
# Supplementary experiments that run after the main pipeline (scripts/run_all.sh). Run from the repository root.
#   bash scripts/run_supplements.sh E14      simulation-matched pairs, hyp evaluation, recall@k and Fig. 1 exports
#   bash scripts/run_supplements.sh E15      largest encoders (recurrent width 1024, transformer width 512)
#   bash scripts/run_supplements.sh E1c      additional widths, same three-phase protocol
#   bash scripts/run_supplements.sh E16      per-pair predictions of the main models (decision analysis)
#   bash scripts/run_supplements.sh E17      certified ABC rewrites: build ABC, regenerate and check the variants, evaluate
# Each experiment is resumable: finished steps leave a marker in results/.done_* and are skipped on rerun.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
EXP=${1:?usage: bash scripts/run_supplements.sh E14|E15|E1c|E16|E17}
GPUS=${GPUS:-0}; PER_GPU=${PER_GPU:-2}; WORKERS=${WORKERS:-$(nproc)}
mkdir -p logs jobs results
Q="python -m tools.run_queue --gpus $GPUS --threads 2 --per-gpu $PER_GPU"
fail() { echo; echo "FAILED: $*"; exit 1; }
step() { local n=$1; shift
  if [ -f results/.done_$n ]; then echo "== $n: already done"; return 0; fi
  echo "== $n: $(date '+%F %T')"; "$@" || fail "step $n (see jobs/FAILED*.txt and <run>/job.log)"; touch results/.done_$n; }
three_phase() {  # lr -> selection -> loss-weight grid -> selection -> final seeds
  local e=$1; export QUEUE_TAG=_$e
  step ${e}_lr       bash -c "python -m tools.plan $e lr && $Q jobs/${e}_lr.jsonl"
  step ${e}_lr_sel   python -m tools.select $e lr
  step ${e}_grid     bash -c "python -m tools.plan $e grid && $Q jobs/${e}_grid.jsonl"
  step ${e}_grid_sel python -m tools.select $e grid
  step ${e}_final    bash -c "python -m tools.plan $e final && $Q jobs/${e}_final.jsonl"
}
need_main() { [ -f runs/E1a/selection.json ] || fail "runs/E1a is missing: run scripts/run_all.sh (at least up to E1a) first"; }

case $EXP in
  E14)
    need_main; export QUEUE_TAG=_E14
    step E14_all bash -c "python -m tools.plan E14 all && $Q jobs/E14_all.jsonl" ;;
  E15|E1c)
    three_phase $EXP ;;
  E16)
    need_main
    for m in base_lr0.002 aug_lr0.002 ttsim_ph_l0.03_lr0.002 circ_ph_l0.2_lr0.002 eqalign_ph_l0.2_lr0.002; do
      for s in 0 1 2 3 4; do
        r=runs/E1a/gru_d64_r2/${m}_s$s
        [ -f $r/pairpred.npz ] && { echo "skip $r"; continue; }
        THREADS=4 python -m tools.dump_pair_preds --run $r || fail "$r"
      done
    done ;;
  E17)
    need_main; export QUEUE_TAG=_E17
    [ -x third_party/abc/abc ] || fail "third_party/abc/abc is missing: run scripts/setup_third_party.sh abc"
    [ -f data/abc.pkl ] || fail "data/abc.pkl is missing"
    step E17_regen python -m tools.check_abc --workers $WORKERS        # rebuilds the ABC variants and compares them with data/abc.pkl
    step E17_eval  bash -c "python -m tools.plan E17 && $Q jobs/E17_all.jsonl"
    step E17_agg   python -m tools.aggregate_e17 ;;
  *) fail "unknown experiment $EXP" ;;
esac
echo "$EXP done $(date '+%F %T')"
