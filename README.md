# Genesis Autonomous Learning

**A runnable, standard-library-only research harness for learning from zero pretrained weights.**

Genesis v0.5 implements an autonomous experimental loop, not a claim of AGI or open-ended general intelligence. The learner starts without models or skills, samples synthetic tasks, invents finite compositions of given operators by search, scores them on disjoint held-out inputs, saves verified compositions as reusable skills, and adjusts the next task difficulty using observed learning progress. A separate small world demonstrates online transition learning and model-based planning.

## Quickstart (Python >=3.11; no packages or API keys)

```bash
python -m unittest discover -s tests -v
python -m genesis train --episodes 100 --seed 42 --budget 300 --output results/run1
python -m genesis paired --episodes 100 --tasks 100 --seed 42 --budget 300 --output results/paired
python -m genesis multiseed --runs 5 --episodes 150 --tasks 100 --budget 300 --output results/replications
python -m genesis bench --episodes 100 --seed 42 --budget 300 --output results/online_ablation
python -m genesis crossdomain --runs 5 --episodes 150 --tasks 100 --budget 300 --output results/crossdomain
python -m genesis discover --runs 5 --episodes 150 --tasks 100 --budget 300 --output results/discovery
python -m genesis emerge --runs 30 --train-steps 2600 --test-steps 1200 --seed 0 --output results/emergence
python -m genesis partial --runs 20 --episodes 30 --steps 250 --max-depth 3 --seed 0 --output results/partial_v01
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

`genesis.partial` learns **short-history predictive probability vectors** in stochastic, partially observable synthetic worlds without hidden-state labels, shared domain encodings or a known state count. It validates memory depth on separate trajectories, compares against memoryless and independently scrambled-history controls, and abstains from adding memory when it does not help. Two opaque discrete token formats are supported; this is not raw image or audio perception. See [partial-observation protocol](research/PARTIAL_OBSERVABILITY.md) and [paired evidence](evidence/partial_v01/summary.json).

`genesis.emergence` **induces new finite predictive-state symbols** by clustering opaque observations with matching action-conditioned future supports. It infers the number of states, learns a compact transition graph, and matches graph structure across separately randomized environments without shared observation/action codebooks, known value codecs, or a supplied state alignment. The evaluator deliberately creates finite isomorphic graphs for its positive regime and unrelated graphs for a negative control. This is restricted representation induction under **strong declared inductive biases** (discrete observations, action labels, repeatability, stationary first-order dynamics and deterministic hidden transitions), not learning without *any* prior assumptions. See [the experiment and negative controls](research/EMERGENCE.md) and [the actual discovered state model](evidence/emergence_v01/invented_representation.json).

`genesis.discovery` infers an opaque operator-index mapping from observed input/output probes under an **explicit shared byte-to-hex anchor convention**, and abstains on incompatible unrelated string operations; see [discovery protocol](research/DISCOVERY.md). `genesis.crossdomain` adds an independently typed string-transformation language with held-out tasks and an explicit operator-index alignment falsification control; the current correspondence is **hand-assigned**, not learned. `genesis.program` has six safe finite 8-bit primitives; candidate programs are data tuples, never executed as Python source. `genesis.synth` finds a minimal-cost match under a strict candidate budget and reuses learned macros as single search operators. `genesis.curriculum` schedules 1–4 primitive-depth problems. `genesis.archive` retains high-scoring behavior descriptors. `genesis.world` implements an empirical transition model and BFS controller. `genesis.rl` implements actual tabular Q-learning with randomized initial behavior and an optional discrete-observation Gymnasium adapter; it does **not** implement DreamerV3 or CleanRL itself.

## Evidence and boundaries

- `partial` learns predictive memory under stochastic transitions and partially observed hidden states. In 20 seeds each, mean heldout NLL fell from 0.986574 to 0.938754 for opaque strings and 0.986021 to 0.940068 for opaque tuple tokens. On 40 IID-hidden-reset negative controls the learner selected zero-depth and no gain; this remains an engineered context-tree estimator on synthetic data, **not** prior-free representation invention. See [protocol and limitations](research/PARTIAL_OBSERVABILITY.md).
- `train` reports attempted/solved tasks, withheld success, candidate evaluation count, skill library size, archive occupancy and difficulty coverage. Seeded runs are reproducible.
- `paired` trains skills, then freezes them and compares adaptive versus no-skill search on **identical unseen tasks** at equal per-task candidate budgets. This isolates the value of the learned library better than `bench`, but remains within one hand-designed task family. `multiseed` adds independent seeds and reports the between-seed variation; independently chosen domains remain necessary.
- `bench` reports a separate online ablation with unpaired adaptive curricula; do not use it alone as evidence of transfer.
- `emerge` reconstructs a **new 7-state symbolic quotient** from 21 unrelated opaque observation tokens in both worlds across 30 seeds, discovers graph isomorphisms (30/30), and rejects 30/30 unrelated control worlds. Mean held-out next-symbol negative log-likelihood: 1.176 induced, 1.317 raw-token, 2.131 shuffled. These synthetic worlds are deliberately identifiable and finite; zero-inductive-bias representation discovery is not achieved. See [methods](research/EMERGENCE.md), [model](evidence/emergence_v01/invented_representation.json), and [results](evidence/emergence_v01/summary.json).
- `crossdomain` freezes numeric skills, compares on a **distinct string task family** and scores a shuffled-index negative control. Five-seed results: +13.0 pp on deliberately aligned motifs, but only +2.4 pp on independently generated motifs (vs +2.0 pp for shuffled). This does **not** establish discovered semantic transfer. [Full protocol and results](research/CROSSDOMAIN.md).
- `discover` tests observation-derived mappings against shuffled and no-transfer controls. Five seeds recovered all six opaque correspondences (5/5); related-motif success was 95.4% vs 75.4% baseline and 80.8% shuffled, but independent-motif gains were small (86.6% vs 84.0% and 85.6%). The known byte-to-hex anchors and a shared teacher-defined operator family make this a restricted mapping-identification result, **not** discovery of semantics from unrelated representations. See [protocol](research/DISCOVERY.md) and [evidence](evidence/discovery_v01/multi_seed_summary.json).
- The built-in problem family is deliberately restricted to a hand-designed 8-bit DSL and a small finite-state world. The system does not acquire general language, unbounded algorithmic creativity, awareness, or unsupervised world knowledge.
- No external source code is copied. [Research comparison and integration roadmap](research/SOURCES.md) documents upstream inspirations. [Benchmark results](research/BENCHMARKS.md) and [reproduction evidence](evidence/multi_seed_5x100.json) are included.

## Next milestones

1. **Now tested:** predictive-history representations under stochastic dynamics, noisy hidden-state observations, and two opaque token formats. Next: learn continuous perceptual embeddings from actual spatial/image/sound streams rather than treating each sample as an atomic symbol.
2. Test nonstationarity, unbounded dependence, out-of-distribution environments and genuinely unrelated transfer with matched compute budgets and failure analyses.
3. Replace the fixed history estimator with optional self-supervised recurrent latent world models and measure long-horizon planning.
4. Implement independently invented and compressive learned operators with holdout verification, not just storing successful operator sequences.
5. Add Gymnasium/Minigrid, OpenSpiel and bsuite adapters behind optional dependency groups before considering sandboxed code mutation with approval gates.

## Security

No network access, no arbitrary source execution, no self-editing or external commands are performed. Library files are parsed as data and validated against the finite DSL. If evolving unrestricted code in future, require an externally enforced sandbox, immutable tests, resource caps, provenance and approval gates; an in-process Python check is not a security boundary.
