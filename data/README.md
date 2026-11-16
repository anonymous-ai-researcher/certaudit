# The certified audit set

All files are Python pickles (plus one JSON). Check them with `python scripts/verify_data.py --summary`; sizes and MD5 sums are in `MANIFEST.json`.

## Format

A **circuit** is a dictionary:

| Key | Meaning |
|:--|:--|
| `n_pi` | number of primary inputs |
| `fanins` | integer array `[n_gates, 2]` of AIG literals: node index × 2 + complement bit. Node 0 is the constant, nodes 1 to `n_pi` are the inputs, AND gates follow in topological order |
| `pos` | output literals |
| `prob` | signal probability of every node, from 4,096 random input patterns shared by a circuit and all its variants |
| `lev` | logic level of every node (longest path from an input) |
| `src` | the design the cone was cut from, for example `arithmetic/log2` |

A **variant record** is a dictionary with the rewritten circuit `var` and the SAT-proven gate pairs: `I` (gate of the original), `J` (gate of the variant) and `P` (polarity: 0 if the two gates compute the same function, 1 if complements). Every variant passed an output-level equivalence check without a conflict limit, and every pair in `I, J, P` is an UNSAT proof.

## Files

| File | Structure |
|:--|:--|
| `data.pkl` | `train`: training circuits; `pairs`: one certified variant per training circuit with its proven pairs; `test`: the 376 test circuits; `tvars`: for each test circuit, the fixed families `S1` to `S4` and injection at rates 0.1, 0.3, 0.5 |
| `heldout.pkl` | for each test circuit, `H1` and `H2` (two circuits have none: the don't-care pass fails on them) |
| `adv.pkl` | for each test circuit, 16 sampled compositions of synthesis and injection, each at most 30% larger |
| `abc.pkl`, `abc.stats.json` | ABC variants (resyn2, dc2, compress2rs, 6-input LUT round trip) of the EPFL and ISCAS test circuits, and the statistics of their generation |
| `iscas.pkl` | the ISCAS85/89 suite in the same layout as the EPFL test side |
| `data_hyp.pkl`, `heldout_hyp.pkl` | the confirmatory design |
| `full_variants.pkl` | full designs and certified variants for the SAT-sweeping probe |
| `pairs_sig64.pkl`, `pairs_sig4096.pkl` | training pairs matched by simulation signatures alone |
| `examples.pkl` | candidate example pairs for Fig. 1 |

## Rewrite families

| Family | Definition | Used for |
|:--|:--|:--|
| S1 | cut rewriting; resubstitution; balancing; SOP refactoring | train, test |
| S2 | balancing; cut rewriting (zero gain); SOP refactoring (zero gain); resubstitution (10 leaves, 3 inserts) | train, test |
| S3 | SOP refactoring; cut rewriting; cut rewriting (zero gain) | train, test |
| S4 | resubstitution (10 leaves, 3 inserts); balancing; cut rewriting; resubstitution | train, test |
| I.r | each gate rebuilt with probability r by one of three identities | train (r = 0.15, 0.3), test (r = 0.1, 0.3, 0.5) |
| H1 | don't-care resubstitution; ESOP balancing; don't-care cut rewriting; don't-care SOP refactoring | test only |
| H2 | each gate rebuilt with probability 0.3 as (x ⊕ y) ⊕ (x ∨ y), each XOR from three ANDs | test only |
| Adversarial | random compositions of the passes above and injection, at most 30% growth | test only |
| ABC | resyn2, dc2, compress2rs, `if -K 6; mfs2; strash` | test only |

## Provenance

The cones are cut from the EPFL combinational benchmark suite and from ISCAS85/89 (combinational cores for ISCAS89). Synthesis passes come from the EPFL logic synthesis libraries through `aigverse 0.1.7`; ABC variants from ABC at commit `f34d3b2`. The generators are `core/certdata.py`, `tools/gen_extra_variants.py`, `tools/gen_abc_variants.py` and `tools/gen_large.py`; all use fixed seeds.
