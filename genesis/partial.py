"""Learn probabilistic predictive history states from opaque POMDP trajectories.

The learner sees action/observation sequences only. It is never given the
simulator's hidden states, its transition/emission probabilities, state count,
semantics of the tokens, or a cross-world correspondence. It discovers a
validated, variable-order predictive representation, not a general POMDP model.

The finite history-depth cap, stationary dynamics, discrete repeatable symbols,
and conditional frequency/backoff estimator ARE hand-designed inductive biases.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
from random import Random
from statistics import mean
from typing import Hashable, Sequence

Token = Hashable
Context = tuple[Token, ...]


@dataclass(frozen=True)
class Episode:
    observations: tuple[Token, ...]
    actions: tuple[Token, ...]

    def __post_init__(self) -> None:
        if not self.actions or len(self.observations) != len(self.actions) + 1:
            raise ValueError('An episode needs at least one action and one extra observation')
        try:
            for v in self.actions + self.observations:
                hash(v)
        except TypeError as e:
            raise ValueError('Opaque tokens must be hashable') from e


@dataclass(frozen=True)
class HiddenWorld:
    """Evaluator-only synthetic generator. Do not pass this to the learner."""
    observation_tokens: tuple[Token, ...]
    action_tokens: tuple[Token, ...]
    transitions: tuple[tuple[tuple[float, ...], ...], ...]
    emissions: tuple[tuple[float, ...], ...]
    reset_each_step: bool = False

    def rollout(self, seed: int, episodes: int, steps: int) -> tuple[Episode, ...]:
        if episodes < 1 or steps < 2:
            raise ValueError('At least one episode of >=2 steps required')
        rng = Random(seed)
        results = []
        n = len(self.emissions)
        for _ in range(episodes):
            state = rng.randrange(n)
            observations = [rng.choices(self.observation_tokens, weights=self.emissions[state])[0]]
            actions = []
            for _ in range(steps):
                ai = rng.randrange(len(self.action_tokens))
                actions.append(self.action_tokens[ai])
                if self.reset_each_step:
                    state = rng.randrange(n)
                else:
                    state = rng.choices(range(n), weights=self.transitions[state][ai])[0]
                observations.append(rng.choices(self.observation_tokens, weights=self.emissions[state])[0])
            results.append(Episode(tuple(observations), tuple(actions)))
        return tuple(results)


def _opaque_tokens(rng: Random, count: int, modality: str, occupied: set) -> tuple[Token, ...]:
    out = []
    while len(out) < count:
        if modality == 'hex':
            token = ''.join(rng.choice('0123456789abcdef') for _ in range(19))
        elif modality == 'vector':
            token = tuple(rng.randrange(256) for _ in range(4))
        else:
            raise ValueError('Modality must be hex or vector')
        if token not in occupied:
            occupied.add(token)
            out.append(token)
    return tuple(out)


def make_world(seed: int, modality: str = 'hex', negative: bool = False) -> HiddenWorld:
    """Unique stochastic graph per seed; no paired isomorphic environments."""
    rng = Random(seed)
    occupied: set = set()
    obs = _opaque_tokens(rng, 3, modality, occupied)
    # Actions do not encode transition semantics; they are randomized labels.
    acts = _opaque_tokens(rng, 2, modality, occupied)
    n = 3
    trans = []
    for state in range(n):
        action_rows = []
        for ai in range(2):
            # Nontrivial stochastic controlled transitions; action 0 persists,
            # action 1 usually rotates. Each seed has independently sampled weights.
            primary = state if ai == 0 else (state + 1) % n
            probs = [rng.uniform(.015, .045) for _ in range(n)]
            probs[primary] += rng.uniform(.78, .91)
            scale = sum(probs)
            action_rows.append(tuple(p / scale for p in probs))
        trans.append(tuple(action_rows))
    emission = []
    for state in range(n):
        probs = [rng.uniform(.08, .18) for _ in range(n)]
        probs[state] += rng.uniform(.57, .72)
        z = sum(probs)
        emission.append(tuple(x / z for x in probs))
    return HiddenWorld(obs, acts, tuple(trans), tuple(emission), negative)


def records(episodes: Sequence[Episode], max_depth: int = 3) -> list[tuple[Context, Token]]:
    """Each record contains opaque present and past; padding never spans episodes."""
    if not 0 <= max_depth <= 6:
        raise ValueError('Depth must be within 0..6')
    data = []
    for episode in episodes:
        for t, current_action in enumerate(episode.actions):
            context: list[Token] = [current_action, episode.observations[t]]
            for d in range(1, max_depth + 1):
                if t >= d:
                    context.extend((episode.actions[t - d], episode.observations[t - d]))
                else:
                    # Unique boundary marker is a private modeling convention.
                    context.extend((None, None))
            data.append((tuple(context), episode.observations[t + 1]))
    return data


def scramble_histories(data: Sequence[tuple[Context, Token]], seed: int) -> list[tuple[Context, Token]]:
    """Negative control: keep current action/observation, randomize older history."""
    rng = Random(seed)
    old = [ctx[2:] for ctx, _ in data]
    rng.shuffle(old)
    return [((ctx[:2] + suffix), target) for (ctx, target), suffix in zip(data, old)]


class PredictiveModel:
    """Interpolated action-conditioned context tree with empirical Bayes backoff."""

    def __init__(self, depth: int, alpha: float = 20.0):
        if depth < 0 or depth > 6 or alpha <= 0:
            raise ValueError('Invalid context depth or smoothing')
        self.depth = depth
        self.alpha = alpha
        self.vocab: tuple[Token, ...] = ()
        self.tables: list[dict[Context, Counter]] = []
        self.prior: dict[Token, float] = {}

    def fit(self, examples: Sequence[tuple[Context, Token]]) -> 'PredictiveModel':
        if not examples:
            raise ValueError('Need observable transitions for fitting')
        self.vocab = tuple(sorted({y for _, y in examples}, key=repr))
        histogram = Counter(y for _, y in examples)
        self.prior = {y: (histogram[y] + 1.0) / (len(examples) + len(self.vocab))
                      for y in self.vocab}
        self.tables = [{} for _ in range(self.depth + 1)]
        for ctx, y in examples:
            if len(ctx) < 2 * (self.depth + 1):
                raise ValueError('Insufficient context features')
            for d in range(self.depth + 1):
                key = ctx[:2 + 2 * d]
                tally = self.tables[d].setdefault(key, Counter())
                tally[y] += 1
        return self

    def probabilities(self, context: Context) -> dict[Token, float]:
        probs = self.prior
        for d, table in enumerate(self.tables):
            tally = table.get(context[:2 + 2 * d], Counter())
            n = tally.total()
            if n:
                probs = {y: (tally[y] + self.alpha * probs[y]) / (n + self.alpha)
                         for y in self.vocab}
        return probs

    def loss(self, examples: Sequence[tuple[Context, Token]]) -> float:
        if not examples:
            raise ValueError('Cannot score an empty holdout')
        total = 0.0
        for ctx, y in examples:
            if y not in self.prior:
                raise ValueError('Test contained unseen output symbol')
            total -= log(self.probabilities(ctx)[y])
        return total / len(examples)

    def representation(self, minimum_support: int = 12, precision: int = 10) -> dict:
        """Export learned predictive signatures, not latent simulator states.

        A signature groups observed contexts with similar conditional predictions.
        Quantization is solely diagnostic, never chosen using test results.
        """
        if minimum_support < 1 or precision < 2:
            raise ValueError('Invalid signature thresholds')
        signatures: dict[tuple[int, ...], list[Context]] = {}
        for ctx, tally in self.tables[self.depth].items():
            if tally.total() < minimum_support:
                continue
            p = self.probabilities(ctx)
            signature = tuple(round(precision * p[y]) for y in self.vocab)
            signatures.setdefault(signature, []).append(ctx)
        return {
            'history_depth': self.depth,
            'predictive_state_count': len(signatures),
            'eligible_context_count': sum(len(x) for x in signatures.values()),
            'invented_predictive_states': [
                {'label': f'P{i}', 'signature': list(sig), 'contexts': len(contexts)}
                for i, (sig, contexts) in enumerate(sorted(signatures.items()))],
            'scope': 'Quantized next-observation distributions over finite recent histories; not hidden-state recovery',
        }


def train_select(train: Sequence[tuple[Context, Token]],
                 validation: Sequence[tuple[Context, Token]],
                 max_depth: int = 3) -> tuple[PredictiveModel, list[float]]:
    """Choose history horizon on validation alone. Never use final test samples."""
    if not 0 <= max_depth <= 6:
        raise ValueError('Invalid max depth')
    candidate = [PredictiveModel(d).fit(train) for d in range(max_depth + 1)]
    scores = [model.loss(validation) for model in candidate]
    # Conservatively prefer the shorter horizon for near-ties (<0.003 nat).
    best = min(range(len(scores)), key=lambda i: (scores[i], i))
    for i in range(best):
        if scores[i] <= scores[best] + .003:
            best = i
            break
    return candidate[best], scores


def trial(seed: int, modality: str = 'hex', negative: bool = False,
          training_episodes: int = 30, steps: int = 250, max_depth: int = 3) -> dict:
    if training_episodes < 2 or steps < 20:
        raise ValueError('Too little training data for the benchmark')
    world = make_world(seed, modality, negative)
    train = records(world.rollout(seed ^ 0x11111, training_episodes, steps), max_depth)
    validation = records(world.rollout(seed ^ 0x22222, max(4, training_episodes // 5), steps), max_depth)
    test = records(world.rollout(seed ^ 0x33333, max(4, training_episodes // 5), steps), max_depth)
    baseline = PredictiveModel(0).fit(train)
    model, scores = train_select(train, validation, max_depth)
    corrupted_train = scramble_histories(train, seed ^ 0x44444)
    corrupted_validation = scramble_histories(validation, seed ^ 0x55555)
    corrupted_test = scramble_histories(test, seed ^ 0x66666)
    shuffled, shuffled_validation = train_select(corrupted_train, corrupted_validation, max_depth)
    # Corrupted model is scored against corrupted test histories; the current
    # action/observation remains intact. All arms use identical test targets.
    raw_loss = baseline.loss(test)
    learned_loss = model.loss(test)
    shuffled_loss = shuffled.loss(corrupted_test)
    repr_summary = model.representation()
    return {
        'seed': seed, 'modality': modality, 'control_world': 'iid-hidden-reset' if negative else 'history-dependent',
        'train_samples': len(train), 'validation_samples': len(validation), 'test_samples': len(test),
        'selected_history_depth': model.depth, 'shuffled_selected_depth': shuffled.depth,
        'validation_loss_by_depth': [round(x, 6) for x in scores],
        'shuffled_validation_loss_by_depth': [round(x, 6) for x in shuffled_validation],
        'raw_nll': round(raw_loss, 6), 'history_nll': round(learned_loss, 6),
        'shuffled_history_nll': round(shuffled_loss, 6),
        'gain_vs_raw': round(raw_loss - learned_loss, 6),
        'gain_vs_shuffled': round(shuffled_loss - learned_loss, 6),
        'representation': repr_summary,
        'caveat': 'Predictive history features and finite depth search are designed in advance; only conditional distributions and selected depth are learned.',
    }


def _mean_bootstrap_ci(values: Sequence[float], seed: int = 91, replicates: int = 2000) -> list[float]:
    """Simple seed-level percentile bootstrap; descriptive, not independent replication."""
    rng = Random(seed)
    n = len(values)
    estimates = sorted(mean(values[rng.randrange(n)] for _ in range(n))
                       for _ in range(replicates))
    return [round(estimates[int(.025 * replicates)], 6),
            round(estimates[int(.975 * replicates)], 6)]


def multiseed(seed: int = 0, runs: int = 10, training_episodes: int = 30,
              steps: int = 250, max_depth: int = 3) -> dict:
    if not 2 <= runs <= 100:
        raise ValueError('runs must be 2..100')
    results = [trial(seed + i, modality, negative, training_episodes, steps, max_depth)
               for modality in ('hex', 'vector') for negative in (False, True)
               for i in range(runs)]
    groups = {}
    for modality in ('hex', 'vector'):
        for negative in (False, True):
            entries = [r for r in results if r['modality'] == modality and
                       (r['control_world'] == 'iid-hidden-reset') == negative]
            groups[f'{modality}:{"iid-control" if negative else "memory"}'] = {
                'runs': len(entries),
                'mean_raw_nll': round(mean(x['raw_nll'] for x in entries), 6),
                'mean_history_nll': round(mean(x['history_nll'] for x in entries), 6),
                'mean_shuffled_nll': round(mean(x['shuffled_history_nll'] for x in entries), 6),
                'mean_gain_vs_raw': round(mean(x['gain_vs_raw'] for x in entries), 6),
                'bootstrap_95_ci_gain_vs_raw': _mean_bootstrap_ci([x['gain_vs_raw'] for x in entries]),
                'mean_gain_vs_shuffled': round(mean(x['gain_vs_shuffled'] for x in entries), 6),
                'history_depth_positive': sum(r['selected_history_depth'] > 0 for r in entries),
                'improved_over_raw': sum(r['gain_vs_raw'] > 0 for r in entries),
                'improved_over_shuffled': sum(r['gain_vs_shuffled'] > 0 for r in entries),
            }
    return {
        'runs_per_condition': runs, 'total_experiments': len(results),
        'training_episodes': training_episodes, 'steps_per_episode': steps,
        'max_history_depth': max_depth, 'groups': groups, 'per_seed': results,
        'interpretation': 'Bounded predictive-memory learning in synthetic POMDPs; no open-world, multimodal semantic understanding, or AGI claim.',
    }
