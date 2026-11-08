# Genesis v0.3: Inferred operator correspondence using observed responses

## Claim being tested

Can a learner with source-domain byte-transformation skills identify six **reordered** target actions by their input/output behavior, without receiving a source-to-target operator-index lookup, and then reuse its skills on previously unseen strings?

**Boundary:** Target strings contain canonical hexadecimal encodings of bytes; source and target share **supplied single-byte value anchors** (`x` ↔ `f'{x:02x}'`). The target transformations were designed to be elementwise lifts of the six known byte primitives. Recovering the hidden permutation under these assumptions is a constrained system-identification task. It is NOT discovery of a representation bridge from unrelated, unanchored modalities.

## Frozen experimental protocol

```bash
python -m unittest discover -s tests -v
python -m genesis discover --runs 5 --episodes 150 --tasks 100 --budget 300 --seed 0 --output evidence/discovery_v01
```

For each seed 0–4:

1. Train with 150 procedural numeric tasks; no pretrained weights, network, model API or code execution.
2. Freeze the 2–8-step source macro library. The evaluator instantiates a new target world with a non-identity pseudorandom permutation of its six opaque action IDs. The learner sees only `apply(action_id, encoded_input) → encoded_output`, not the permutation or task-generating programs.
3. Use eight single-byte calibration anchors and eight disjoint confirmation anchors drawn from 0–255. Compare target responses against source-operator responses under the **known** hex codec. Accept a mapping only when all six matches are unique and exact on calibration **and** confirmation. Record the full cost (96 target-oracle queries per successful seed). Freeze this mapping **before task evaluation**.
4. As a mismatch check, attempt the identical algorithm on the original independently designed six string operations (`prefix_a`, `drop_first`, `suffix_b`, `reverse`, `rotate_left`, `swap_ab`). Incompatible domains must cause abstention, not a fabricated correspondence. No evaluation tasks are attempted in this unrelated domain.
5. Draw 100 new 2–4-byte strings per seed for each of two evaluation regimes. The strings are longer than the 1-byte calibration anchors, and examples are disjoint from the 24 sealed final inputs within each task. `related` draws from the same **teacher-supplied structural motifs** used to generate numeric training tasks with probability 0.65; `independent` draws uniformly from target action sequences.
6. Evaluate each task using identical input/output examples and independently initialized single-use judges for: `baseline` (no macros), `learned` (source macros projected using the inferred mapping), and `shuffled` (the exact same macros after a nonidentity mapping shuffle). All three have at most 300 candidate evaluations and at most 9,600 primitive/example applications per task; the costs of mapping probes and numeric training are reported **separately**, not counted into each target-task search budget.
7. Final holdout accuracies are report-only, never used to infer mappings, promote skills, tune policy or choose a candidate. No source code is executed by candidates; programs are finite tuples.

## Results (five seeds, 1,000 unique evaluation tasks, 3 tested arms per task)

| Regime | No transfer | Inferred mapping | Shuffled mapping | Inferred − baseline | Inferred − shuffled |
|---|---:|---:|---:|---:|---:|
| Teacher-aligned motifs (500 tasks) | 75.4% | 95.4% | 80.8% | +20.0 pp | +14.6 pp |
| Independently sampled motifs (500 tasks) | 84.0% | 86.6% | 85.6% | +2.6 pp | +1.0 pp |

All five hidden target operator permutations were inferred and independently confirmed. The original unrelated string operators caused abstention in all five seeds. Per-seed inferred-vs-baseline differences: related `[+0.20, +0.16, +0.20, +0.25, +0.19]`; independent `[+0.03, +0.04, +0.01, +0.02, +0.03]`. Source skill-library sizes were `[33, 27, 32, 30, 33]`. A total of **96 calibration and confirmation calls per seed** were needed to identify the target action permutation, on top of source training cost.

The aligned result shows **successful operator identification and template reuse under a shared mathematical substrate**. The near-parity of learned and shuffled templates on independent motifs is evidence **against** claiming generic transferable reasoning. These tasks are synthetic and correlated, and five seeds are not an independent replication or proof of a generally applicable method. The current evaluation also does not match end-to-end elapsed time across methods, and the calibration procedure presupposes the known hex codec and an exact isomorphism.

Artifacts: `evidence/discovery_v01/multi_seed_summary.json` and `seed_*/{summary.json,trials.jsonl}`. Raw trials allow computation of alternative statistics without rerunning. No mutable live data or external dependencies are required.

## Further falsification targets

- Learn a usable representation map when no common byte-value anchors or known codec are supplied, from relational or compositional constraints alone.
- Insert target operators that are only approximately related, non-bijective, or actively perturbed; reject mappings without false acceptance. Calibrate confidence and model uncertainty.
- Test on independently authored domains (not lifted byte operators) and adopt fixed-budget, wall-clock, training-cost and data-efficiency accounting. Include multiple random permutations, different source skill distributions and stronger search baselines.
- Keep identification (calibration) separate from transfer (held-out tasks) and avoid tuning on the final challenge set.
