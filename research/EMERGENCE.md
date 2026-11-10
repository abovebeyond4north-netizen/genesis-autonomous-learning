# Genesis v0.4 — Opaque predictive-state representation induction

## Operational claim

Genesis creates NEW predictive-state symbols from action-conditioned interaction tuples without supplying hidden state count, value codecs, shared tokens, action names, a cross-world index alignment, or pretrained networks.

This is **not** literal learning with zero priors. The learning rule assumes discrete repeatable symbols, identified actions, stationary first-order dynamics, deterministic hidden transitions, and sufficient coverage. The evaluator deliberately creates finite isomorphic positive-world graphs; this is an experimental prior, not autonomous invention of open-ended intelligence.

## Input firewall and experiment

The induction function receives only raw triples (current_observation, opaque_action, next_observation) from random-action trajectories. The benchmark evaluator (never the learner) generates:
- A random strongly connected 7-state graph, with 3 actions and 3 independently sampled observation aliases per state.
- A positive paired world with exactly the same underlying graph but independently permuted hidden state IDs, opaque action IDs, and a completely disjoint observation codebook.
- An unrelated random graph as the falsification control.

Each identifier is a random 15–23-character token. No source-to-target token correspondence or action mapping is passed to the learner.

## Induction and falsification

1. Obtain 2,600 action-conditioned transitions from each world. The collector follows random trajectories, not latent-state-selected i.i.d. sampling.
2. Abstain when any observed observation/action pair has fewer than 16 samples.
3. Create equivalence classes grouping tokens with identical observed next-token support **for every discovered action**. The class count is inferred, not supplied.
4. Give the classes new symbols S0, S1, ... and learn the quotient transition table. Reject any class whose observed transitions lead to inconsistent successor classes.
5. Infer a cross-world graph isomorphism by searching both state and action permutations. No shared codebook, common value anchors or supplied symbol alignment is used. Exact search is bounded to seven states and three actions.
6. Freeze the representation before a separately seeded 1,200-transition held-out trajectory. Compare next-token negative log likelihood with the same Laplace smoothing for raw-token contexts, induced-state contexts and size-preserving randomly shuffled classes.
7. Only evaluator-side partition scoring uses hidden state labels; induction never receives them.

Reproduce with Python 3.11+, no API keys or packages:

    python -m unittest discover -s tests -v
    python -m genesis emerge --runs 30 --train-steps 2600 \
      --test-steps 1200 --seed 0 --output results/emergence

The command writes summary.json (including all 30 per-seed observations) and invented_representation.json (actual discovered observation classes, learned quotient graphs, action labels, and structural alignment).

## 30-seed result

| Metric | Observed |
|---|---:|
| Exact recovered partitions in both worlds, evaluator-only scoring | 30 / 30 |
| Isomorphic world pair aligned without token/action identity anchors | 30 / 30 |
| Independently generated unrelated graph rejected | 30 / 30 |
| Opaque observation tokens compressed into newly induced states | 21 → 7 |
| Heldout raw-token next-token NLL | 1.316902 nats |
| Heldout induced-state NLL | 1.176142 nats |
| Heldout shuffled-state NLL | 2.131102 nats |
| Induced improvement over raw | 0.140759 nats |
| Induced improvement over shuffled | 0.954960 nats |

Actual seed-0 induced quotient (columns are ordered by the newly observed opaque action IDs):

| Invented symbol | Action 0 | Action 1 | Action 2 |
|---|---|---|---|
| S0 | S3 | S6 | S5 |
| S1 | S6 | S2 | S4 |
| S2 | S0 | S3 | S6 |
| S3 | S2 | S5 | S2 |
| S4 | S1 | S4 | S3 |
| S5 | S4 | S1 | S0 |
| S6 | S5 | S0 | S1 |

These S-labels are newly constructed group IDs, **not** hidden labels provided by the benchmark. Their numbering is arbitrary.

## What has not been established

- When given only 1,000 transitions per world, the support-coverage rule abstained on all 30 exploratory seeds. This is a low-data weakness.
- The estimator fails outside its model assumptions: stochastic hidden transitions, partial observability, nonstationarity, continuous or unrepeatable inputs, unknown temporal boundaries, etc.
- Positive pairs are graph-isomorphic **by benchmark design**; related structures are discovered, not generated from unrelated domains.
- A finite-state/Markov prior and explicit notion of predictive support are hardcoded. Learning a fully novel objective or arbitrary representations without assumptions is not demonstrated.
- Baselines are raw context lookup and shuffled partitions, not state-of-the-art predictive reconstruction or MDL; 30 pseudorandom seeds are not independent external replication.

Related independent literature includes transCSSR (https://github.com/ddarmon/transCSSR), deep bisimulation for control (https://rowanmcallister.github.io/publication/dbc/) and theory of state abstraction (https://ojs.aaai.org/index.php/AAAI/article/view/5075).

Next step: independent, partially observable worlds without guaranteed isomorphism; stronger predictive-state and compression baselines; matched costs, explicit false-abstraction rates and frozen third-party evaluations.
