# Cross-domain structural-transfer falsification — October 31, 2026

## Hypothesis

Does a library of program *structures* learned while solving 8-bit integer transformations
reduce search cost or improve generalization on a **different string-transformation task**?
This is distinct from genuine invention of a shared semantic representation: operator
indices are aligned **by the researcher**, not inferred from experience.

## Design

```bash
python -m unittest discover -s tests -v
python -m genesis crossdomain --runs 5 --episodes 150 --tasks 100 --budget 300 --seed 0 --output evidence/crossdomain_v01
```

For each seed:

1. Train with 150 self-selected numeric DSL challenges, initially with zero skills and no weights.
2. Freeze the resulting byte-domain skill library and project each learned sequence to integer **operator indices**.
3. Produce two **unseen string-task regimes** with six different operators over variable-length text:
   - `aligned`: 65% of targets reuse a **hand-designated structural motif distribution** also favored during numeric training; the indices are assigned by the researcher.
   - `independent`: operator sequences are uniform random and have **no shared target motif generator**.
4. Compare three methods on identical per-task visible examples and sealed final holdouts:
   - `baseline`: no learned templates;
   - `adaptive`: frozen numeric skill templates mapped by index to the string operators;
   - `shuffled`: same learned templates after a deterministic shuffled index mapping (negative control).
5. Bound each method by **300 distinct candidate evaluations and 9,600 primitive/example applications** per task. All holdouts are hidden from candidate selection; the score is measured once after search.

Each regime has 100 heldout tasks per seed, 500 total. The source learner sees no
string-domain examples or labels. There is no model pretraining or external service.

## Observed results

| String regime | Baseline final pass | Transferred pass | Shuffled pass | Transfer vs baseline | Transfer primitive work saving |
|---|---:|---:|---:|---:|---:|
| Researcher-aligned motifs | 80.2% | 93.2% | 84.0% | **+13.0 pp** | 35.43% |
| Independently generated motifs | 87.4% | 89.8% | 89.4% | **+2.4 pp** | 0.11% |

Five independent seeds; each seed contributes 100 tasks per regime. Seed-to-seed
standard deviation of transfer effect: 4.53 pp aligned; 2.30 pp independent.
The shuffled control gains 3.8 pp and 2.0 pp respectively over baseline.
On independently generated tasks, the difference between transferred and shuffled
is only **0.4 pp**, so there is no convincing evidence that semantically meaningful
knowledge transferred rather than a generic search bias. By contrast, the aligned
motif condition demonstrates that an explicitly shared index structure *can* be
reused, which is an engineering proof of mechanism, not evidence of autonomous
alignment discovery.

Raw per-task results are under `evidence/crossdomain_v01/seed_*/trials.jsonl`;
full per-seed metrics are under `seed_*/summary.json` and in
`evidence/crossdomain_v01/multi_seed_summary.json`.

## Limitations and next falsification

- The new task family is still small, synthetic, and hand-designed. A literal index
  alignment is given, and the aligned target generator uses matching index motifs.
- Search comparisons share candidate and primitive-application caps but do not
  establish matched wall-clock time or comparable engineering effort.
- Five seeds and correlated tasks do not justify a broad statistical claim;
  the confidence intervals are exploratory, not externally replicated.
- The `StringJudge` hides targets from the solver API but is not a malicious-code
  security boundary; this benchmark does not run arbitrary generated Python.
- The higher aligned score is **not** evidence for autonomous discovery of
  representations, learned cross-domain semantic mapping or open-ended intelligence.

**Next experiment:** infer an operator correspondence from independently collected
input/output observations (without shared indices), freeze it before evaluation,
then compare against a shuffled mapping, a no-transfer baseline, and unrelated
string tasks. Reject inferred mappings that fail holdouts or only help tasks whose
latent generator was constructed to match training.
