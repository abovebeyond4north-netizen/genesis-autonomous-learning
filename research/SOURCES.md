# SOTA architecture survey (2026-10-31)

This is a **curated** survey of compatible techniques, not an assertion that every repository or paper has been inspected or that there is a single SOTA metric for open-ended intelligence.

| Codebase | Capability | Actual Genesis v0.1 implementation | Status |
| --- | --- | --- | --- |
| [DreamerV3](https://github.com/danijar/dreamerv3) | Latent world models, imagined actor-critic rollouts | Empirical, discrete transition model and BFS planner | **Analogy only**, not DreamerV3/JAX |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | Reproducible RL baselines, seeded experiments | Deterministic seeds, event logs, ablation CLI | Reimplemented concept |
| [pyribs](https://github.com/icaros-usc/pyribs) | QD archives / MAP-Elites | Coarse behavioral occupancy archive | Inspired concept, not CMA-ME |
| [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) | Population search and verified program evolution | Finite program enumeration with persistent reusable macros | Inspired concept; **no** LLM mutation |
| [Voyager](https://github.com/MineDojo/Voyager) | Executable skill libraries, curriculum, repair from feedback | Data-only composable macros and learning-progress curriculum | Inspired concept; **no** Minecraft/LLM |
| [AI-Scientist-v3](https://github.com/SakanaAI/AI-Scientist-v3) | Research/ablation/final testing envelope | Single-use validation and final test partitions | Partial protocol |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | Multiagent testbeds | None yet | Candidate optional adapter |
| [Minigrid](https://github.com/Farama-Foundation/Minigrid) | Procedural environment benchmarks | Small custom finite-state environment | Candidate optional adapter |
| [QDax](https://github.com/adaptive-intelligent-robotics/QDax) | Scalable quality-diversity search | None | Candidate JAX accelerator |
| [Jumanji](https://github.com/instadeepai/jumanji) | Diverse JAX puzzles and environments | None | Candidate evaluation suite |
| [bsuite](https://github.com/google-deepmind/bsuite) | RL diagnostic experiments | None | Candidate eval suite |
| [DCD](https://github.com/facebookresearch/dcd) | Robust unsupervised environment design, PLR and PAIRED | Learning-progress depth selection only | Conceptual; no adversarial level design |
| [Darwin Gödel Machine](https://github.com/kew-lab/darwin-godel-machine) | Iterative benchmarked agent-source mutation | No arbitrary code mutation | Candidate sandboxed extension, needs pretrained LLM |
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | LLM-guided evolving algorithm search | Bounded deterministic program enumeration | Candidate optional evolution engine |
| [A-Evolve](https://github.com/A-EVO-Lab/a-evolve) | Search over agent skill code/workspaces | Data-only skill library | Candidate future benchmark plugin |

## Integration policy and licensing

These additional projects are integration candidates, not implementations bundled into Genesis.

Genesis is independently written; **no source code or weights from upstream projects were imported**. That avoids accidental relicensing or unsafe dependency coupling. Upstream licensing and API compatibility must be checked at the pinned commit before direct inclusion. The project does not claim to replicate benchmark performance or architecture of any upstream project.

## Research constraints

- There is **no pretrained model**, network access, cloud API, or external dependency in the initial core.
- "From scratch" means the search states are initially empty and experience updates them. Primitive operators, procedural task distribution, evaluator, observation encoding and planner were designed by humans.
- Successful synthesis on a bounded byte-transform DSL is not evidence of unbounded intelligence, novel axioms, consciousness, or algorithm invention independent of designed primitives.
- Avoid target leakage: only eight examples are presented to synthesis; validation and final partitions are private, disjoint and one-shot.
- `paired` evaluates on frozen unseen tasks with an equal search budget, and `bench` separately measures online learning with task-distribution confounds. Neither demonstrates transfer to a fundamentally new domain.
- The archive is QD-inspired, not an exact implementation of MAP-Elites or pyribs.
