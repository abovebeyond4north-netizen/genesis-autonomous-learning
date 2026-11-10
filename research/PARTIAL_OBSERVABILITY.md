# Genesis v0.5: History-derived predictive representations in stochastic POMDPs

## Narrow question

Can a learner starting with no pretrained weights infer useful **predictive history states**
from action/observation sequences when observations are ambiguous and transitions are
stochastic? Does additional memory improve unseen next-observation prediction compared
with a memoryless model and a destroyed-history control? Does it decline to use history
when historical information is causally useless?

This does **not** establish open-ended invention, recovery of hidden simulator states,
object understanding, general-purpose perception, or learning without inductive bias.

## Experiment and information boundary

- The evaluator independently generates a stochastic controlled process for every seed:
  three unreported hidden states; two randomly labeled actions; two strictly positive,
  nondegenerate action-conditioned transition distributions; and overlapping noisy
  emission distributions. The learner never receives any of these arrays, or the state
  number. No two generated worlds are assumed or required to be isomorphic.
- Observations are opaque, randomly generated **19-character strings** or **4-integer
  tuples**. They represent different symbolic token formats, not raw vision/audio.
  Action names are also opaque. The learner operates only on equality/hashability.
- For each world, **30 separate 250-step training episodes**, six 250-step validation
  episodes, and six distinct 250-step sealed test episodes use different RNG streams.
- The learner estimates next-observation distributions from a bounded recent context:
  the present action/observation plus up to three preceding action/observation pairs.
  Its predictive vector is a learned representation; near-identical vectors are
  quantized into diagnostic predictive classes, **not** claimed as recovered latent
  physical states. Context length is selected solely using validation log loss.
- A smoothed backoff estimator shares probability mass with shorter contexts; no
  generated source code is executed. The horizon 0–3, Bayes-style smoothing and
  predictive-state definition are researcher-designed assumptions.
- Three matched next-observation tests use identical target episodes: learned history,
  memoryless present-action/current-observation prediction, and history randomized
  independently within each split while preserving present action/observation.
- A separate negative-control world **resets the hidden state independently before
  every step**, so recent history should have no predictive advantage.
- Twenty seeds per modality and condition: **80 experiments**, each with 1,500 heldout
  transitions. The per-condition seed-level percentile bootstrap intervals are
  descriptive; 20 related synthetic seeds are not externally independent replication.

## Observations

| Condition (20 seeds each) | Memoryless NLL | Learned history NLL | Randomized history NLL | Mean gain (nats) | 95% bootstrap interval |
|---|---:|---:|---:|---:|---|
| Opaque strings, memory matters | 0.986574 | **0.938754** | 0.986574 | 0.047820 | [0.043062, 0.052752] |
| Opaque tuples, memory matters | 0.986021 | **0.940068** | 0.986021 | 0.045953 | [0.039950, 0.052170] |
| Opaque strings, reset control | 1.098655 | 1.098655 | 1.098655 | 0.000000 | [0.0, 0.0] |
| Opaque tuples, reset control | 1.098767 | 1.098767 | 1.098767 | 0.000000 | [0.0, 0.0] |

- History was selected for all **40/40 memory-dependent** runs, and yielded lower
  heldout NLL in all of them. The IID-reset worlds selected no history in **40/40**
  cases; their equal losses are partly by design, because all three arms fall back
  to the same zero-depth estimator.
- The learned predictor reduces average NLL by ~4.8% (strings) and ~4.7% (tuples)
  relative to the corresponding memoryless models on these synthetic worlds.
- The randomly permuted older-history arm selected memoryless predictions in all
  observed runs. That is a useful falsification but **not** a computationally matched
  neural control or a test of physical causal understanding.
- The learned quantized signatures are empirical groupings of short histories. Their
  arbitrary labels and number do not have a guaranteed one-to-one relationship with
  the unobserved simulator states.

## Reproduce

```bash
python -m unittest discover -s tests -v
python -m genesis partial --runs 20 --episodes 30 --steps 250 --max-depth 3 --seed 0 --output results/partial_v01
```

`evidence/partial_v01/summary.json` includes *compacted* paired seed metrics, negative
controls and a fully expanded illustrative predictive signature snapshot for the
first seed. Rerunning the command exports full per-context signatures for every run
in its un-compacted `summary.json` and `trials.jsonl` files. Validation and test
trajectories are independent within each synthetic world; the generator and random
seeds are fully specified in source. All benchmark evidence is report-only and is
not used to train a new model in the same run.

## Falsification / next tasks

1. Replace the categorical equality-only observations with genuine images, sound,
   or spatial signals and learn observation embeddings rather than a fixed token
   vocabulary; evaluate on task distributions not produced by this generator.
2. Train and test on nonstationary dynamics, long-range memory, action-dependent
   observation selection, and switched mechanisms, reporting the failures as well
   as positive gains.
3. Compare against hidden Markov models, recurrent state-space models and
   information-bottleneck baselines at matched wall-clock and train data budgets.
4. Infer a portable model shared by related but not guaranteed-isomorphic worlds
   without user-supplied observation anchors, then test transfer on unrelated worlds.
