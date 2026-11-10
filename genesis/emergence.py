"""Induce latent predictive states from opaque action-conditioned observations.

No labels, operator index mapping, state counts, value codec, pretrained model
or shared cross-world tokens are passed to the induction algorithm. The GENERAL
priors remain: discrete repeatable observations/actions, stationary first-order
action-conditioned dynamics, and deterministic hidden transitions. This is not
prior-free open-ended intelligence.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations, combinations
from math import log
from random import Random
from statistics import mean
from typing import Iterable

Triple = tuple[str, str, str]


class InsufficientEvidence(ValueError):
    """Do not invent states when observation coverage is inadequate."""


@dataclass(frozen=True)
class OpaqueWorld:
    """Benchmark-only hidden dynamics. The learner receives triples, never this."""
    table: tuple[tuple[int, ...], ...]
    actions: tuple[str, ...]
    aliases: tuple[tuple[str, ...], ...]

    def interact(self, observation: str, action: str, rng: Random) -> str:
        try:
            state = next(i for i, codes in enumerate(self.aliases) if observation in codes)
            action_index = self.actions.index(action)
        except (StopIteration, ValueError) as exc:
            raise ValueError("Unknown observation or action") from exc
        return rng.choice(self.aliases[self.table[state][action_index]])


def _token(rng: Random, used: set[str]) -> str:
    while True:
        result = "".join(rng.choice("0123456789abcdef") for _ in range(rng.randrange(15, 24)))
        if result not in used:
            used.add(result)
            return result


def _graph(rng: Random, states: int, actions: int) -> tuple[tuple[int, ...], ...]:
    for _ in range(1000):
        columns = []
        for _ in range(actions):
            col = list(range(states))
            rng.shuffle(col)
            columns.append(col)
        graph = tuple(tuple(c[s] for c in columns) for s in range(states))
        reached, agenda = {0}, [0]
        while agenda:
            s = agenda.pop()
            for t in graph[s]:
                if t not in reached:
                    reached.add(t)
                    agenda.append(t)
        if len(reached) == states:
            return graph
    raise RuntimeError("Could not generate connected graph")


def _world(graph: tuple[tuple[int, ...], ...], rng: Random,
           aliases: int, used: set[str]) -> OpaqueWorld:
    actions = tuple(_token(rng, used) for _ in range(len(graph[0])))
    observations = tuple(tuple(_token(rng, used) for _ in range(aliases))
                         for _ in graph)
    return OpaqueWorld(graph, actions, observations)


def make_worlds(seed: int, states: int = 7, actions: int = 3,
                aliases_per_state: int = 3) -> tuple[OpaqueWorld, OpaqueWorld, OpaqueWorld]:
    """Two independently encoded isomorphic graphs, plus an unrelated control."""
    if not (3 <= states <= 7 and 2 <= actions <= 3 and 2 <= aliases_per_state <= 5):
        raise ValueError("Supported: states 3..7, actions 2..3, aliases 2..5")
    rng, used = Random(seed), set()
    graph = _graph(rng, states, actions)
    a = _world(graph, rng, aliases_per_state, used)
    state_order, action_order = list(range(states)), list(range(actions))
    rng.shuffle(state_order)
    rng.shuffle(action_order)
    relabeled = [None] * states
    for old, new in enumerate(state_order):
        relabeled[new] = tuple(state_order[graph[old][old_a]]
                               for old_a in action_order)
    b = _world(tuple(relabeled), rng, aliases_per_state, used)
    independent = _world(_graph(rng, states, actions), rng, aliases_per_state, used)
    return a, b, independent


def observe(world: OpaqueWorld, count: int, seed: int) -> tuple[Triple, ...]:
    """Random-action trajectory, without balanced latent-state sampling."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = Random(seed)
    current = rng.choice(rng.choice(world.aliases))
    trace = []
    for _ in range(count):
        action = rng.choice(world.actions)
        next_obs = world.interact(current, action, rng)
        trace.append((current, action, next_obs))
        current = next_obs
    return tuple(trace)


@dataclass(frozen=True)
class Representation:
    actions: tuple[str, ...]
    classes: tuple[tuple[str, ...], ...]
    transition_table: tuple[tuple[int, ...], ...]

    @property
    def index(self) -> dict[str, int]:
        return {token: i for i, group in enumerate(self.classes) for token in group}

    @property
    def symbol_count(self) -> int:
        return sum(len(g) for g in self.classes)


def induce(transitions: Iterable[Triple], min_samples_per_condition: int = 16) -> Representation:
    """Infer unknown number of states by action-conditioned predictive support.

    Empirical one-step support equality is valid in this finite deterministic
    hidden-state benchmark, not arbitrary stochastic or non-Markov processes.
    """
    if min_samples_per_condition < 1:
        raise ValueError("Minimum support must be positive")
    samples = tuple(transitions)
    if not samples:
        raise InsufficientEvidence("No interaction data")
    actions = tuple(sorted({a for _, a, _ in samples}))
    observations = tuple(sorted({x for x, _, _ in samples} | {y for _, _, y in samples}))
    counts: Counter[tuple[str, str]] = Counter()
    supports: dict[tuple[str, str], set[str]] = defaultdict(set)
    for x, a, y in samples:
        counts[x, a] += 1
        supports[x, a].add(y)
    if any(counts[x, a] < min_samples_per_condition
           for x in observations for a in actions):
        raise InsufficientEvidence("Unobserved or under-sampled observation/action condition")
    signatures: dict[tuple[frozenset[str], ...], list[str]] = defaultdict(list)
    for x in observations:
        signatures[tuple(frozenset(supports[x, a]) for a in actions)].append(x)
    classes = tuple(sorted(tuple(sorted(group)) for group in signatures.values()))
    index = {token: i for i, group in enumerate(classes) for token in group}
    table = []
    for group in classes:
        row = []
        for action in actions:
            targets = {index[y] for x in group for y in supports[x, action]}
            if len(targets) != 1:
                raise InsufficientEvidence("Transitions violate deterministic quotient")
            row.append(next(iter(targets)))
        table.append(tuple(row))
    return Representation(actions, classes, tuple(table))


def discover_alignment(left: Representation, right: Representation) -> dict | None:
    """Infer both state and action correspondence by exact graph isomorphism."""
    n, m = len(left.classes), len(left.actions)
    if n < 2 or n != len(right.classes) or m != len(right.actions):
        return None
    if n > 7 or m > 3:
        raise ValueError("Exact search is bounded to 7 states/3 actions")
    for action_map in permutations(range(m)):
        for state_map in permutations(range(n)):
            if all(state_map[left.transition_table[s][a]]
                   == right.transition_table[state_map[s]][action_map[a]]
                   for s in range(n) for a in range(m)):
                return {"states": dict(enumerate(state_map)),
                        "actions": dict(zip(left.actions,
                                            (right.actions[j] for j in action_map)))}
    return None


def alignment_consistency(left: Representation, right: Representation, alignment: dict) -> float:
    state_map = alignment["states"]
    action_map = alignment["actions"]
    right_actions = {a: i for i, a in enumerate(right.actions)}
    valid = total = 0
    for s in range(len(left.classes)):
        for ai, action in enumerate(left.actions):
            total += 1
            mapped = state_map.get(s)
            ja = right_actions.get(action_map.get(action))
            if mapped is not None and ja is not None and (
                state_map.get(left.transition_table[s][ai]) == right.transition_table[mapped][ja]
            ):
                valid += 1
    return valid / total if total else 0.0


def prediction_loss(train: tuple[Triple, ...], test: tuple[Triple, ...],
                    representation: Representation | None, alpha: float = .5) -> float:
    """Disjoint evaluation: smoothed next-observation negative log-likelihood."""
    if alpha <= 0 or not test:
        raise ValueError("Nonzero smoothing and test data are required")
    vocab = {x for x, _, _ in train} | {y for _, _, y in train}
    if not vocab or any(y not in vocab for _, _, y in test):
        raise InsufficientEvidence("Heldout observation outside training vocabulary")
    labels = representation.index if representation else {}
    if representation and (vocab - labels.keys()):
        raise InsufficientEvidence("Representation missing training observations")
    counts: Counter[tuple[object, str, str]] = Counter()
    totals: Counter[tuple[object, str]] = Counter()
    for x, a, y in train:
        k = labels[x] if representation else x
        counts[k, a, y] += 1
        totals[k, a] += 1
    loss = 0.0
    for x, a, y in test:
        k = labels.get(x) if representation else x
        if k is None:
            raise InsufficientEvidence("Heldout source observation unknown")
        loss -= log((counts[k, a, y] + alpha) /
                    (totals[k, a] + alpha * len(vocab)))
    return loss / len(test)


def shuffled_representation(rep: Representation, seed: int) -> Representation:
    """Shuffled equivalence partition with identical class sizes."""
    rng = Random(seed)
    tokens = [x for group in rep.classes for x in group]
    rng.shuffle(tokens)
    offset, classes = 0, []
    for group in rep.classes:
        classes.append(tuple(tokens[offset:offset + len(group)]))
        offset += len(group)
    return Representation(rep.actions, tuple(classes), rep.transition_table)


def partition_f1(rep: Representation, world: OpaqueWorld) -> float:
    """Evaluator-only latent label check; never called by learner."""
    actual = {c: i for i, g in enumerate(world.aliases) for c in g}
    predicted = rep.index
    if set(actual) != set(predicted):
        return 0.0
    tp = fp = fn = 0
    for a, b in combinations(actual, 2):
        truth, guess = actual[a] == actual[b], predicted[a] == predicted[b]
        tp += truth and guess
        fp += not truth and guess
        fn += truth and not guess
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0


def representation_snapshot(seed: int = 0, train_steps: int = 2600) -> dict:
    """Serialize actual learner-created latent states; no hidden graph labels."""
    first, second, _ = make_worlds(seed)
    a = induce(observe(first, train_steps, seed ^ 0xabc113))
    b = induce(observe(second, train_steps, seed ^ 0xdef123))
    alignment = discover_alignment(a, b)
    return {"seed": seed,
            "source": {"invented_state_symbols": [f"S{i}" for i in range(len(a.classes))],
                       "classes_of_opaque_observations": a.classes,
                       "action_symbols": a.actions,
                       "learned_transition_table": a.transition_table},
            "target": {"invented_state_symbols": [f"T{i}" for i in range(len(b.classes))],
                       "classes_of_opaque_observations": b.classes,
                       "action_symbols": b.actions,
                       "learned_transition_table": b.transition_table},
            "discovered_graph_isomorphism": alignment,
            "notice": "Only opaque (observation, action, next_observation) triples were supplied."}


def single_trial(seed: int, train_steps: int = 2600, test_steps: int = 1200) -> dict:
    a, b, unrelated = make_worlds(seed)
    tr_a = observe(a, train_steps, seed ^ 0xabc113)
    tr_b = observe(b, train_steps, seed ^ 0xdef123)
    tr_c = observe(unrelated, train_steps, seed ^ 0xabc133)
    te_b = observe(b, test_steps, seed ^ 0x15A5B1)
    left, right, negative = (induce(t) for t in (tr_a, tr_b, tr_c))
    found = discover_alignment(left, right)
    independent = discover_alignment(left, negative)
    raw_loss = prediction_loss(tr_b, te_b, None)
    learned_loss = prediction_loss(tr_b, te_b, right)
    shuffled_loss = prediction_loss(tr_b, te_b, shuffled_representation(right, seed ^ 0xdeadbeef))
    return {"seed": seed, "samples_per_world": train_steps,
            "heldout_samples": test_steps,
            "raw_symbols_left": left.symbol_count,
            "induced_states_left": len(left.classes),
            "raw_symbols_right": right.symbol_count,
            "induced_states_right": len(right.classes),
            "evaluator_partition_f1_left": round(partition_f1(left, a), 6),
            "evaluator_partition_f1_right": round(partition_f1(right, b), 6),
            "matched_isomorphism": found is not None,
            "independent_world_rejected": independent is None,
            "match_consistency": alignment_consistency(left, right, found) if found else 0.0,
            "raw_heldout_nll": round(raw_loss, 6),
            "induced_heldout_nll": round(learned_loss, 6),
            "shuffled_heldout_nll": round(shuffled_loss, 6),
            "loss_improvement_vs_raw": round(raw_loss - learned_loss, 6),
            "loss_improvement_vs_shuffled": round(shuffled_loss - learned_loss, 6),
            "states_in_generator_evaluator_only": len(a.table),
            "caveat": ("Worlds are constructed to be finite, Markov and deterministic under alias noise. "
                       "One positive graph pair is isomorphic by construction. Learner gets no hidden "
                       "state labels, state count, value codec, shared action tokens, or correspondence.")}


def multiseed(start_seed: int = 0, runs: int = 30,
              train_steps: int = 2600, test_steps: int = 1200) -> dict:
    if not 2 <= runs <= 100:
        raise ValueError("runs must be 2..100")
    if train_steps < 1 or test_steps < 1:
        raise ValueError("Data counts must be positive")
    records = [single_trial(start_seed + i, train_steps, test_steps) for i in range(runs)]
    return {"runs": runs, "start_seed": start_seed,
            "train_steps_per_world": train_steps,
            "heldout_steps_per_world": test_steps,
            "matched_graphs": sum(r["matched_isomorphism"] for r in records),
            "exact_partitions_both_worlds": sum(
                r["evaluator_partition_f1_left"] == 1.0 and
                r["evaluator_partition_f1_right"] == 1.0 for r in records),
            "unrelated_graphs_rejected": sum(r["independent_world_rejected"] for r in records),
            "mean_raw_nll": round(mean(r["raw_heldout_nll"] for r in records), 6),
            "mean_induced_nll": round(mean(r["induced_heldout_nll"] for r in records), 6),
            "mean_shuffled_nll": round(mean(r["shuffled_heldout_nll"] for r in records), 6),
            "mean_nll_gain_vs_raw": round(mean(r["loss_improvement_vs_raw"] for r in records), 6),
            "mean_nll_gain_vs_shuffled": round(mean(r["loss_improvement_vs_shuffled"] for r in records), 6),
            "per_seed": records,
            "interpretation": ("Narrow predictive-state abstraction in finite synthetic worlds. "
                               "Not prior-free general intelligence.")}