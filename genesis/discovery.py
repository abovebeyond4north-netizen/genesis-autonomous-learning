"""Observation-based operator correspondence in an encoded string world.

This experiment intentionally supplies shared byte-value anchors, not an operator
index map. The target vocabulary is independently shuffled every seed, and the
learner must discover a bijection from black-box input/output queries before
seeing any evaluation tasks. Unrelated string operations must cause abstention.

It is NOT evidence for discovering representations without shared anchors.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from typing import Callable
import json
import random

from .program import PRIMITIVES, transform
from .skills import SkillLibrary
from .trainer import run_training
from .crossdomain import STRING_OPS, string_transform


class OpaqueHexWorld:
    """Opaque permutation of six byte operators acting on encoded byte strings.

    The oracle exposes apply(index, string); correspondence is deliberately not
    exposed to infer_mapping. Numeric and target IDs are independently ordered.
    """

    def __init__(self, target_to_source: tuple[int, ...]):
        if (len(target_to_source) != len(PRIMITIVES)
                or set(target_to_source) != set(range(len(PRIMITIVES)))):
            raise ValueError('Expected a bijection over six byte operators')
        self.__mapping = target_to_source

    @property
    def size(self) -> int:
        return len(PRIMITIVES)

    def apply(self, op_index: int, value: str) -> str:
        if type(op_index) is not int or not 0 <= op_index < self.size:
            raise ValueError('Unknown action')
        if (not isinstance(value, str) or len(value) < 2 or len(value) > 16
                or len(value) % 2 or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Expected a canonical nonempty lowercase hex byte string')
        op = PRIMITIVES[self.__mapping[op_index]]
        return bytes(transform(op, x) for x in bytes.fromhex(value)).hex()

    def target_index(self, source_index: int) -> int:
        """Teacher-only task generation, never supplied to mapping inference."""
        return self.__mapping.index(source_index)


def infer_mapping(
    target_oracle: Callable[[int, str], str],
    n_target_ops: int,
    calibration: tuple[int, ...],
    confirmation: tuple[int, ...],
) -> dict:
    """Infer source->target mapping from I/O; accept only unique exact matches.

    Target action identifiers are opaque. No target task labels, heldout scores,
    task motifs or true permutation can be passed to this function. The source
    byte input values are known anchors shared by both representations.
    """
    if (n_target_ops != len(PRIMITIVES) or not calibration or not confirmation
            or len(set(calibration)) != len(calibration)
            or len(set(confirmation)) != len(confirmation)
            or set(calibration) & set(confirmation)
            or any(type(x) is not int or not 0 <= x < 256 for x in calibration + confirmation)):
        raise ValueError('Malformed or overlapping anchors')
    queried = 0
    def observed(xs: tuple[int, ...]) -> tuple[tuple[str, ...], ...]:
        nonlocal queried
        rows = []
        for t in range(n_target_ops):
            row = []
            for x in xs:
                queried += 1
                try:
                    result = target_oracle(t, f'{x:02x}')
                except (ValueError, TypeError, IndexError):
                    result = None
                row.append(result)
            rows.append(tuple(row))
        return tuple(rows)
    calibrations = observed(calibration)
    expected = tuple(tuple(f'{transform(op, x):02x}' for x in calibration)
                     for op in PRIMITIVES)
    matches = {s: [t for t, row in enumerate(calibrations) if row == expected[s]]
               for s in range(len(PRIMITIVES))}
    # In particular, reject accidental duplicates and noninjective matches.
    unique = all(len(ts) == 1 for ts in matches.values())
    mapping = tuple(matches[s][0] for s in range(len(PRIMITIVES))) if unique else ()
    unique = unique and len(set(mapping)) == n_target_ops
    verified = False
    if unique:
        confirmed = observed(confirmation)
        verified = all(confirmed[mapping[s]] == tuple(
            f'{transform(op, x):02x}' for x in confirmation)
            for s, op in enumerate(PRIMITIVES))
    return {'mapping': list(mapping) if unique and verified else None,
            'abstained': not (unique and verified),
            'oracle_queries': queried,
            'calibration_anchors': len(calibration),
            'confirmation_anchors': len(confirmation),
            'rejected_ambiguous_or_incompatible': not unique,
            'rejected_confirmation': unique and not verified}


def execute(world: OpaqueHexWorld, program: tuple[int, ...], x: str) -> str:
    for i in program:
        x = world.apply(i, x)
    return x


@dataclass(frozen=True)
class HexChallenge:
    examples: tuple[tuple[str, str], ...]
    max_depth: int
    task_id: int


class HexJudge:
    def __init__(self, world: OpaqueHexWorld, target: tuple[int, ...], final: tuple[str, ...]):
        self.__world, self.__target, self.__final = world, target, final
        self.__used = False

    def final_accuracy(self, candidate: tuple[int, ...] | None) -> float:
        if self.__used:
            raise RuntimeError('Sealed evaluator is single-use')
        self.__used = True
        if candidate is None:
            return 0.0
        return sum(execute(self.__world, candidate, x) == execute(self.__world, self.__target, x)
                   for x in self.__final) / len(self.__final)


def make_task(world: OpaqueHexWorld, rng: random.Random, depth: int,
              task_id: int, regime: str) -> tuple[HexChallenge, HexJudge]:
    if depth not in (2, 3, 4) or regime not in ('related', 'independent'):
        raise ValueError('Unknown task configuration')
    if regime == 'related' and rng.random() < .65:
        # Training and test worlds share a teacher-defined sequence distribution;
        # the learner never sees this choice, only the resulting I/O examples.
        motifs = ((4, 3), (0, 5))   # rotl,xor1 and inc,invert in numeric DSL
        seq: list[int] = []
        while len(seq) < depth:
            seq.extend(rng.choice(motifs))
        source_seq = tuple(seq[:depth])
    else:
        source_seq = tuple(rng.randrange(world.size) for _ in range(depth))
    target = tuple(world.target_index(s) for s in source_seq)
    # Evaluation strings contain 2-4 bytes, unlike single-byte calibration.
    values = set()
    while len(values) < 32:
        value = bytes(rng.randrange(256) for _ in range(rng.choice((2, 3, 4)))).hex()
        values.add(value)
    samples = sorted(values)
    rng.shuffle(samples)
    examples = tuple((x, execute(world, target, x)) for x in samples[:8])
    return HexChallenge(examples, depth, task_id), HexJudge(world, target, tuple(samples[8:]))


@dataclass(frozen=True)
class SearchResult:
    program: tuple[int, ...] | None
    evaluations: int
    primitive_work: int


def solve(world: OpaqueHexWorld, challenge: HexChallenge,
          macros: tuple[tuple[int, ...], ...] = (), budget: int = 300,
          work_budget: int | None = None) -> SearchResult:
    if budget < 1 or (work_budget is not None and work_budget < 1):
        raise ValueError('Invalid search budget')
    if (not challenge.examples or not 1 <= challenge.max_depth <= 4
            or len({x for x, _ in challenge.examples}) != len(challenge.examples)):
        raise ValueError('Invalid examples')
    valid = tuple(dict.fromkeys(m for m in macros if isinstance(m, tuple)
                   and 2 <= len(m) <= challenge.max_depth
                   and all(type(i) is int and 0 <= i < world.size for i in m)))[:128]
    alphabet = tuple((i,) for i in range(world.size)) + valid
    queue: list[tuple[int, int, tuple[int, ...]]] = [(0, 0, ())]
    seen: set[tuple[int, ...]] = set()
    attempts = work = 0
    limit = work_budget if work_budget is not None else budget * len(challenge.examples) * 4
    while queue and attempts < budget:
        cost, length, seq = heappop(queue)
        if seq:
            if seq in seen:
                continue
            seen.add(seq)
            effort = length * len(challenge.examples)
            if effort + work > limit:
                continue
            work += effort
            attempts += 1
            if all(execute(world, seq, x) == y for x, y in challenge.examples):
                return SearchResult(seq, attempts, work)
        if length == challenge.max_depth:
            continue
        for item in alphabet:
            if length + len(item) <= challenge.max_depth:
                heappush(queue, (cost + 1, length + len(item), seq + item))
    return SearchResult(None, attempts, work)


def _summary(records: list[dict], arm: str) -> dict:
    scores = [r[arm + '_pass'] for r in records]
    base = [r['baseline_pass'] for r in records]
    return {'pass_rate': round(mean(scores), 4),
            'delta_vs_baseline': round(mean(scores) - mean(base), 4),
            'only_this_passes': sum(a and not b for a, b in zip(scores, base)),
            'only_baseline_passes': sum(b and not a for a, b in zip(scores, base)),
            'candidate_evaluations': sum(r[arm + '_evaluations'] for r in records),
            'primitive_applications': sum(r[arm + '_work'] for r in records)}


def discovery_transfer(seed: int = 0, train_episodes: int = 100,
                       tasks_per_regime: int = 100, budget: int = 300,
                       output: Path | None = None) -> dict:
    if train_episodes < 1 or tasks_per_regime < 1 or budget < 1:
        raise ValueError('Invalid experiment budget')
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        run_training(train_episodes, seed, budget, directory)
        numeric_skills = SkillLibrary(directory / 'skills.json').macros()
    templates = tuple(tuple(PRIMITIVES.index(op) for op in p.ops) for p in numeric_skills)
    rng = random.Random(seed ^ 0x9B48F1)
    permutation = list(range(len(PRIMITIVES)))
    rng.shuffle(permutation)
    if permutation == list(range(len(PRIMITIVES))):
        permutation = permutation[1:] + permutation[:1]
    world = OpaqueHexWorld(tuple(permutation))
    # Calibration and confirmation anchors are disjoint; no multi-byte evaluation
    # samples or evaluation program outputs are available to inference.
    anchors = random.Random(seed ^ 0xABCDEF).sample(range(256), 16)
    inference = infer_mapping(world.apply, world.size, tuple(anchors[:8]), tuple(anchors[8:]))
    mapping = inference['mapping']
    if mapping is not None:
        projected = tuple(tuple(mapping[i] for i in seq) for seq in templates)
        shuffled_mapping = mapping[:]
        random.Random(seed ^ 0xBADC0DE).shuffle(shuffled_mapping)
        if shuffled_mapping == mapping:
            shuffled_mapping = shuffled_mapping[1:] + shuffled_mapping[:1]
        scrambled = tuple(tuple(shuffled_mapping[i] for i in seq) for seq in templates)
    else:
        projected, scrambled = (), ()
    # Strings without common byte-transform semantics are not force-matched.
    unrelated = infer_mapping(string_transform, len(STRING_OPS),
                              tuple(anchors[:8]), tuple(anchors[8:]))
    result = {'seed': seed, 'train_episodes': train_episodes, 'tasks_per_regime': tasks_per_regime,
              'budget': budget, 'numeric_skills': len(templates),
              'mapping_observations': inference,
              'mapping_correct_after_sealed_teacher_check': (
                  mapping is not None and all(world.target_index(i) == mapping[i]
                                              for i in range(len(PRIMITIVES)))),
              'unrelated_string_mapping_abstained': unrelated['abstained'],
              'unrelated_string_calibration_oracle_queries': unrelated['oracle_queries'],
              'regimes': {},
              'caveat': ('Opaque action mapping is inferred from unlabeled operator responses to '
                         'SHARED single-byte value anchors, not from unanchored observations. '
                         'The related task generator explicitly shares numeric motifs. '
                         'The unrelated native string domain should trigger abstention. '
                         'No open-ended cross-domain semantics are demonstrated.')}
    all_records = []
    eval_rng = random.Random(seed ^ 0x1230C0DE)
    for regime in ('related', 'independent'):
        records = []
        for task_id in range(tasks_per_regime):
            depth = eval_rng.choice((2, 3, 4))
            task_seed = eval_rng.getrandbits(63)
            challenge, judge_learned = make_task(world, random.Random(task_seed), depth, task_id, regime)
            _, judge_base = make_task(world, random.Random(task_seed), depth, task_id, regime)
            _, judge_shuffle = make_task(world, random.Random(task_seed), depth, task_id, regime)
            arms = {'learned': (projected, judge_learned),
                    'baseline': ((), judge_base), 'shuffled': (scrambled, judge_shuffle)}
            rec = {'regime': regime, 'task_id': task_id, 'depth': depth}
            for name, (macros, judge) in arms.items():
                found = solve(world, challenge, macros, budget, budget * 8 * 4)
                rec[name + '_pass'] = judge.final_accuracy(found.program) == 1
                rec[name + '_evaluations'] = found.evaluations
                rec[name + '_work'] = found.primitive_work
            records.append(rec)
            all_records.append(rec)
        result['regimes'][regime] = {
            'tasks': len(records), 'baseline': _summary(records, 'baseline'),
            'learned': _summary(records, 'learned'),
            'shuffled': _summary(records, 'shuffled')}
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        (output / 'trials.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in all_records), encoding='utf-8')
    return result


def multiseed_discovery(start_seed: int = 0, runs: int = 5, train_episodes: int = 100,
                        tasks_per_regime: int = 100, budget: int = 300,
                        output: Path | None = None) -> dict:
    if not 2 <= runs <= 30:
        raise ValueError('runs must be in 2..30')
    per_seed = [discovery_transfer(start_seed + i, train_episodes, tasks_per_regime,
                                  budget, output / f'seed_{start_seed + i}' if output else None)
                for i in range(runs)]
    summary = {'runs': runs, 'start_seed': start_seed, 'episodes': train_episodes,
               'tasks_per_regime_per_seed': tasks_per_regime, 'budget': budget,
               'all_inferred_mappings_confirmed': all(r['mapping_correct_after_sealed_teacher_check']
                                                       for r in per_seed),
               'all_unrelated_string_abstained': all(r['unrelated_string_mapping_abstained']
                                                     for r in per_seed),
               'regimes': {}, 'per_seed': per_seed}
    for regime in ('related', 'independent'):
        learned = [r['regimes'][regime]['learned']['delta_vs_baseline'] for r in per_seed]
        shuffled = [r['regimes'][regime]['shuffled']['delta_vs_baseline'] for r in per_seed]
        summary['regimes'][regime] = {
            'mean_learned_delta': round(mean(learned), 4),
            'sd_learned_delta': round(stdev(learned), 4),
            'mean_shuffled_delta': round(mean(shuffled), 4),
            'sd_shuffled_delta': round(stdev(shuffled), 4),
            'per_seed_learned_delta': learned,
            'per_seed_shuffled_delta': shuffled}
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'multi_seed_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary
