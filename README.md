# Genesis Autonomous Learning

**A runnable, standard-library-only research harness for learning from zero pretrained weights.**

Genesis v0.1 implements an autonomous experimental loop, not a claim of AGI or open-ended general intelligence. The learner starts without models or skills, samples synthetic tasks, invents finite compositions of given operators by search, scores them on disjoint held-out inputs, saves verified compositions as reusable skills, and adjusts the next task difficulty using observed learning progress. A separate small world demonstrates online transition learning and model-based planning.

## Quickstart (Python >=3.11; no packages or API keys)

```bash
python -m unittest discover -s tests -v
python -m genesis train --episodes 100 --seed 42 --budget 300 --output results/run1
python -m genesis paired --episodes 100 --tasks 100 --seed 42 --budget 300 --output results/paired
python -m genesis multiseed --runs 5 --episodes 150 --tasks 100 --budget 300 --output results/replications
python -m genesis bench --episodes 100 --seed 42 --budget 300 --output results/online_ablation
python -m genesis world --episodes 30 --seed 1
python -m genesis rl --episodes 500 --seed 1
# Optional real benchmark:
# pip install gymnasium
# python -m genesis gym --env-id FrozenLake-v1 --episodes 2000
```

The CLI prints JSON summaries. When `--output` is specified, the learner saves `skills.json`, `events.jsonl` and `summary.json` under the run folder. Training resuming is **not** implemented: existing saved skills will be loaded, but curriculum and event counts restart. Use a fresh output folder for comparable runs.

## Core loop

```text
empty initial library
      │
      ▼
learning-progress + UCB difficulty scheduler
      │
      ▼
procedural 8-bit transformation challenge (8 visible examples)
      │
      ▼
weighted program search: primitives + verified compositional macros
      │
      ▼
private one-shot validation (16 distinct inputs)
      │ success
      ├──→ persistent skill library + behavioral diversity archive
      │
      ▼
sealed final score (24 other inputs; report-only)
      │
      ▼
record metrics → next task
```

`genesis.program` has six safe finite 8-bit primitives; candidate programs are data tuples, never executed as Python source. `genesis.synth` finds a minimal-cost match under a strict candidate budget and reuses learned macros as single search operators. `genesis.curriculum` schedules 1–4 primitive-depth problems. `genesis.archive` retains high-scoring behavior descriptors. `genesis.world` implements an empirical transition model and BFS controller. `genesis.rl` implements actual tabular Q-learning with randomized initial behavior and an optional discrete-observation Gymnasium adapter; it does **not** implement DreamerV3 or CleanRL itself.

## Evidence and boundaries

- `train` reports attempted/solved tasks, withheld success, candidate evaluation count, skill library size, archive occupancy and difficulty coverage. Seeded runs are reproducible.
- `paired` trains skills, then freezes them and compares adaptive versus no-skill search on **identical unseen tasks** at equal per-task candidate budgets. This isolates the value of the learned library better than `bench`, but remains within one hand-designed task family. `multiseed` adds independent seeds and reports the between-seed variation; independently chosen domains remain necessary.
- `bench` reports a separate online ablation with unpaired adaptive curricula; do not use it alone as evidence of transfer.
- The built-in problem family is deliberately restricted to a hand-designed 8-bit DSL and a small finite-state world. The system does not acquire general language, unbounded algorithmic creativity, awareness, or unsupervised world knowledge.
- No external source code is copied. [Research comparison and integration roadmap](research/SOURCES.md) documents upstream inspirations. [Benchmark results](research/BENCHMARKS.md) and [reproduction evidence](evidence/multi_seed_5x100.json) are included.

## Next milestones

1. Expand paired heldout evaluation across genuinely different domains and add confidence intervals / matched wall-clock compute baselines.
2. Replace the finite transition table with optional self-supervised latent world models and measure long-horizon planning.
3. Implement independently invented and compressive learned operators with heldout verification, not just storing successful operator sequences.
4. Add Gymnasium/Minigrid, OpenSpiel, and bsuite adapters behind optional dependency groups.
5. Add a sandboxed code mutation plugin with pinned trust boundaries and Git rollback, only after paired benchmarks pass.

## Security

No network access, no arbitrary source execution, no self-editing or external commands are performed. Library files are parsed as data and validated against the finite DSL. If evolving unrestricted code in future, require an externally enforced sandbox, immutable tests, resource caps, provenance and approval gates; an in-process Python check is not a security boundary.
