# Reproducing the paper

This file maps every table, figure and claim of the paper to the command that produces it. Run everything from the repository root.

## 0. Requirements

| | Minimum | What we used |
|:--|:--|:--|
| Python | 3.10 | 3.11 (GPU run), 3.13 (CPU run) |
| PyTorch | 2.x | 2.7.1 (GPU run), 2.15 (CPU run) |
| GPUs | none for the smoke test and the CPU run | 8 GPUs for the full pipeline |
| CPU / memory | 2 cores, 7 GB for the CPU run | 48+ cores, 128+ GB (batching, simulation and SAT run on CPU and are the bottleneck) |
| Disk | 1 GB | 30 to 50 GB for all runs |

```bash
conda create -n certaudit python=3.11 -y && conda activate certaudit
pip install torch --index-url https://download.pytorch.org/whl/cu121     # choose the build for your CUDA version
pip install -r requirements.txt
python scripts/verify_data.py --summary
bash scripts/smoke_test.sh
```

`torch_scatter` and `torch_sparse` are **not** needed: python-deepgate imports but never calls them, and the code falls back to built-in equivalents.

## 1. Main pipeline

```bash
bash scripts/setup_third_party.sh all
python -m tools.env_check                 # every line must be [PASS], last line ENVIRONMENT OK
GPUS=0,1,2,3,4,5,6,7 PER_GPU=3 bash scripts/run_all.sh
```

`PER_GPU` is the number of jobs per GPU: 3 with at least 48 cores and 128 GB, 2 with less, 1 below 64 GB. The script is safe to interrupt and rerun; finished trainings and evaluations are skipped. On 8 GPUs, steps 1 to 6 took roughly 12 to 20 hours and the whole pipeline without `E5` roughly 1.5 to 2.5 days. Order:

| Step | Experiment | Phases |
|:-:|:--|:--|
| 1 | `E0` environment reproduces the CPU run (7 regimes × 3 seeds) + noise floor | all |
| 2 | `E3` released DeepGate2: audit, fine-tuning with its own loss and with EqAlign | grid, final |
| 3 | `E1a` widths 64 and 256, 7 regimes, 5 seeds | lr, grid, final |
| 4 | `E8` mechanism controls | grid, final |
| 5 | `E9` H2 variants added to training | all |
| 6 | `E12` leave one family out | all |
| 7 | `E13` large circuits (builds `data/large.pkl`, 20-minute certification limit per cone; CPU-bound) | data, eval |
| 8 | `E10` DeepGate2 retrained from its released code and data, 3 seeds | all |
| 9 | `E1b`, `E4`, `E2` more widths, data scale, other architectures | lr, grid, final |
| 10 | `E5` SAT-sweeping probe (`RUN_E5=1`; CPU-bound) | all |

Every experiment with phases runs **learning rate → selection → loss-weight grid → selection → final seeds**; both selections use validation circuits only.

Supplements, after the main pipeline:

```bash
bash scripts/run_supplements.sh E14    # simulation-matched pairs; hyp evaluation; recall@k and Fig. 1 exports
bash scripts/run_supplements.sh E15    # recurrent width 1024, transformer width 512
bash scripts/run_supplements.sh E1c    # additional widths
bash scripts/run_supplements.sh E16    # per-pair predictions for the decision analysis
bash scripts/run_supplements.sh E17    # certified ABC rewrites
```

## 2. From runs to numbers

```bash
python -m tools.aggregate E1           # also E0 E3 E8 E9 E12 E13 E4 E2 E10 E5, or "all"
python -m tools.aggregate_e17
```

Each call writes, under `results/<EXP>/`:

- `summary.csv`: one row per (suite, architecture, regime) with seed means, standard deviations and bootstrap intervals of every metric;
- `comparisons.csv`: design-level differences between regimes, with cluster-bootstrap intervals, sign tests and the number of designs on which each side wins;
- `claims.md`: the rules written before the experiment, each marked as holding or failing.

The copies in `results/` of this repository are the outputs of these commands on our runs.

## 3. Paper item by item

| Item | Where it comes from |
|:--|:--|
| **Table I** (main results) | `results/E1/summary.csv`, rows `suite=epfl, arch=gru_d64_r2` |
| **Fig. 2** (audit per family, worst case, level bins) | per-circuit records `runs/E1a/gru_d64_r2/*/raw_epfl.pkl` → `paper_pipeline/make_figs2.py audit` |
| **Fig. 3** (two levels of repair) | `E1a` + `E8` → `paper_pipeline/fig_twolevel.py` |
| **RQ1 decision rules** | `E16` exports → `cpu_run/decisions.py` |
| **RQ1 solver test** (CaDiCaL, 69 instances) | `cpu_run/solver_phase_test/` (build → run → analyze); output in `results/solver_phase_test/summary.json` |
| **RQ2 design-level differences** | `results/E1/comparisons.csv` |
| **RQ2 controls** (frozen encoder, no negatives, no head) | `results/E8/` |
| **RQ2 simulation pairs** | `E14` |
| **RQ2 transfer** (leave one family out, ABC, H2 added) | `results/E12/`, `results/E17/`, `results/E9/` |
| **RQ3, Fig. 4** | `E5` for refuted calls; `cpu_run/sweep.py` for the single-core timings |
| **Table II, ISCAS** | `results/E1/summary.csv`, rows `suite=iscas` |
| **Table II, bidirectional / transformer** | `results/E2/summary.csv` |
| **Table II, released checkpoints** | `cpu_run/audit_external.py` (DeepGate2, DeepGate3 tokenizer, FGNN2, MGVGA, TRACE with their released weights) |
| **Table II, DeepGate2 retrained / fine-tuned** | `results/E10/`, `results/E3/` |
| **RQ4 model size** | `results/E1/`, `results/E15/`, `results/n_params.json` |
| **RQ4 large circuits** | `results/E13/` |
| **RQ4 confirmatory design `hyp`** | `cpu_run/hyp_build.py` then evaluation of the 15 checkpoints in `checkpoints/cpu_run/` |
| **Extended version, appendix tables** | `paper_pipeline/make_appendix.py`, `make_arch_table.py`, `make_scale_table.py`, `make_e4_tex.py`, `make_e17.py` |

## 4. The CPU run

`cpu_run/` is the earlier, complete pipeline on a 2-core CPU (3 seeds). The paper takes from it only the confirmatory design, the released-checkpoint audits, the example of Fig. 1, the timings of Fig. 4b and the solver test. The GPU pipeline reproduces 67 of its 68 comparable metrics within tolerances fixed in advance (`python -m tools.repro_check`, reference values in `configs/paper_reference.json`).

Its scripts are provided as they were used, with paths made relative to a working folder `work/`. They were not re-run from this packaged copy.

## 5. Typesetting the paper

`paper_pipeline/build.sh` lists, in order, the scripts that write `numbers*.tex`, every generated table and every figure from the run outputs, and then compiles the paper. The LaTeX sources with all generated files already in place are in `paper/latex/`:

```bash
cd paper/latex
pdflatex main && bibtex main && pdflatex main && pdflatex main      # 6 pages + references
pdflatex ext  && bibtex ext  && pdflatex ext  && pdflatex ext       # extended version, 26 pages
```

## 6. Determinism

CPU runs are deterministic given the seed. CUDA scatter and index kernels are not bit-deterministic, so GPU reruns differ slightly between runs; all reported numbers are means over 5 seeds with their standard deviations. Data generation uses fixed seeds, and `tools/check_abc.py` rebuilds the ABC variants and verifies that they are identical to `data/abc.pkl`.
