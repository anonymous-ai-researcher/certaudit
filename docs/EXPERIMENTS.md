# Experiments and protocol

## Regimes

| Name in the paper | `--method` | Loss added to the L1 probability loss |
|:--|:--|:--|
| Baseline | `base` | none, original circuits only |
| Augmentation | `aug` | none, one certified variant per circuit |
| TT-sim | `ttsim_ph` | DeepGate2-style: cosine similarity of 64 random gate pairs regressed onto truth-table similarity |
| Circuit-level contrast | `circ_ph` | InfoNCE between the mean projection of a circuit and of its variant |
| L1 alignment | `l1align` | MixGate-style: L1 distance between the states of proven pairs, no negatives |
| L1 alignment + TT-sim | `l1tt` | both of the above |
| **EqAlign** | `eqalign_ph` | symmetric InfoNCE over up to 512 proven pairs per batch, temperature 0.1, on a projection head |
| EqAlign without head | `eqalign` | the same loss on the encoder state |
| Positives-only cosine | `cosalign_ph` | cosine alignment through the same head, no negatives |
| TT-sim with DeepGate2's pair filters | `ttsimf_ph` | control for the simplification of TT-sim |

Architectures (`--arch`): `gru` (recurrent, the main encoder), `bi` (attention and reverse pass, as in DeepGate), `tf` (transformer). `--d` is the state width, `--rounds` the number of passes.

A single run:

```bash
python -m core.train --method eqalign_ph --lam 0.2 --arch gru --d 64 --rounds 2 --lr 0.002 --seed 0 --out runs/demo/eqalign_s0
python -m core.evaluate --run runs/demo/eqalign_s0 --suites val,epfl,iscas
```

`core.evaluate` writes `val.json` (used only for selection), and per-circuit records `raw_epfl.pkl`, `raw_iscas.pkl` (and `raw_abc.pkl`, `raw_epfl_h.pkl` for the suites `abc`, `epfl_h`).

## Experiment list

| ID | Question | Design |
|:--|:--|:--|
| E0 | Does this environment reproduce the CPU run? | width 64, 7 regimes × 3 seeds, same settings; 70 metrics compared with tolerances fixed in advance; seed-to-seed noise floor |
| E1a | Do the claims hold at two widths with a clean selection protocol? | widths 64 and 256, 7 regimes, learning rate and loss weights selected on validation, 5 seeds |
| E1b, E1c, E15 | Does model size remove the blindness? | widths 128 to 1024, 4 rounds at width 256, transformer width 512 |
| E2 | Other architectures | bidirectional encoder (64, 128, 256) and transformer (128, 256) |
| E3 | Released DeepGate2 | audit of the released weights under two primary-input protocols; fine-tuning with its own loss and with EqAlign, 5 seeds |
| E4 | Is it a matter of data? | 1, 2 or 4 certified variants per training circuit, widths 64 and 256 |
| E5 | SAT sweeping | candidate ordering by embeddings, five full designs × four certified variants |
| E8 | Mechanisms | positives-only alignment, EqAlign without head, filtered TT-sim, head trained on a frozen encoder |
| E9 | Is H2 learnable? | XOR-rewrite variants added to training |
| E10 | Is it one checkpoint? | DeepGate2 retrained from its released code and data, 3 seeds |
| E12 | Transfer within synthesis | one family (S1, S2, S3, S4 or injection) removed from training, tested on it |
| E13 | Larger circuits | cones of 503 to 5,497 gates, evaluation only |
| E14 | Do proofs matter for training? | EqAlign trained on pairs matched by 64-bit and 4,096-bit signatures; evaluation on the confirmatory design |
| E16 | Decision consequences | per-pair predictions of the main models |
| E17 | Industrial-strength rewrites | resyn2, dc2, compress2rs and a LUT round trip in ABC, certified by two checkers; evaluation only |

## Selection rule

Fixed before any run and applied identically to every regime (`tools/select.py`):

1. the learning rate is the one with the lowest validation MAE of the **baseline**;
2. each loss weight is the one with the highest restructured Top-1 on validation among those whose validation MAE is within 25% of the baseline;
3. picks at the edge of a grid are reported, not re-tuned;
4. no test circuit is read by any selection step.

## Statistics

Implemented in `core/stats.py` and applied by `tools/aggregate.py`:

- test circuits are cones of 7 designs and overlap within a design, so every comparison is made on **design-level** means;
- 95% cluster-bootstrap intervals over designs, two-sided sign tests (smallest attainable p with 7 designs: 0.016), Holm correction over regimes on the 17 ISCAS designs;
- for the 4 large-circuit designs, Welch tests over seeds with Holm correction.

## Metrics

Computed per circuit and then averaged over the circuits of a family (`core/evaluate.py: pair_metrics`):

| Metric | Definition |
|:--|:--|
| ΔMAE | error on the variant's matched gates minus error on their proven equivalents |
| inconsistency δ | absolute difference between the two predictions of a proven pair (complement-aware) |
| ρ | mean δ over restructured pairs divided by the mean error on the same original gates |
| gap closed g | 1 − δ / δ⊥, where δ⊥ permutes the variant-side errors within the variant |
| Top-1 | the proven equivalent is the unique nearest original gate by cosine similarity; ties count as misses |
| AUROC_lvl | proven equivalent against a random original gate at the same logic level |
| worst case | the adversarial variant with the largest ΔMAE, per circuit |

A pair is *restructured* when the structural hashes of the two fanin cones differ. All primary inputs share one learned initial state, so the recurrent encoder reads the structure of a fanin cone, not which inputs feed it; gates whose cones are isomorphic up to input names tie and count as misses.
