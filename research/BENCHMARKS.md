# Local benchmark evidence — October 31, 2026

All experiments ran locally with Python standard library and randomly initialized learning state; no third-party models, tokens, or cloud calls. The exact summary file is `evidence/multi_seed_5x100.json`.

## Protocol

```bash
python -m unittest discover -s tests -v
python -m genesis multiseed --runs 5 --episodes 150 --tasks 100 --budget 300 --output results/replications
python -m genesis paired --seed 42 --episodes 150 --tasks 100 --budget 300 --output results/seed42
python -m genesis rl --episodes 500 --seed 1
```

Each training run induces a curriculum that selects problem difficulty from 1 through 4 and constructs finite 8-bit transformation programs from preselected primitive operators. The proposed program is fitted using only eight visible input/output examples. Sixteen independent validation probes gate skill promotion. Final probes (24) are never used to select programs or change a candidate in that task. The evaluation split uses 100 additional tasks generated from an independent seed stream; the macro library is frozen during evaluation. Both algorithms receive the same challenges, budgets, and hidden tests.

## Observations

- Five seeds, 150 training episodes per seed, 100 paired evaluation tasks per seed.
- Adaptive heldout passes by seed: 98, 95, 96, 97, 97.
- No-skill heldout passes by seed: 77, 76, 74, 83, 77.
- Mean absolute improvement: **19.2 percentage points**, between-seed standard deviation **3.11 percentage points**.
- Mean relative candidate-evaluation reduction: **63.77%**.
- In an additional independent seed-42 run, pass rates were 88% versus 86%, and candidate evaluations 6,573 versus 10,972.
- Tabular RL in `HiddenDirectionWorld` reduced mean steps per episode from 11.32 (first 100) to 6.56 (last 100) over 500 episodes. Both blocks reached the goal, so steps rather than episode success is the informative metric.
- 12 automated tests passed when the artifacts were packaged.

## Interpretation and limitations

Skill reuse provides empirical advantage **within a deliberately compositional, procedurally generated task family**. The train and heldout sets are not different real-world domains, and they share the same primitive vocabulary and task motifs. In particular, evidence supports shorter search and improved finite-DSL synthesis—not self-invention of all operators, multimodal representation learning, general intelligence or broad open-world capabilities. Task generation itself is developer-specified, as is the independent evaluator. Sampling uncertainty, test multiplicity and domain shift remain open issues. External framework integrations are not benchmarked yet.
