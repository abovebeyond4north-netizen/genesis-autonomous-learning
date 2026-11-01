"""Experimental cross-domain *structural* transfer: byte programs -> string programs.

There is intentionally no shared runtime operator, oracle, embedding or label between
training and testing. An index-based operator alignment is an explicit assumption,
NOT a learned semantic mapping. Shuffled alignment is a falsification control.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
import json
import random

from .program import PRIMITIVES
from .skills import SkillLibrary
from .trainer import run_training


# Each operation is total on all strings, including the empty string.
# The vocabulary, data types and semantics differ from the byte DSL.
STRING_OPS = ("prefix_a", "drop_first", "suffix_b", "reverse", "rotate_left", "swap_ab")
STRING_INPUTS = tuple(''.join(p) for n in range(2, 6) for p in product('abc', repeat=n))


def string_transform(index: int, value: str) -> str:
    if index == 0: return 'a' + value
    if index == 1: return value[1:]
    if index == 2: return value + 'b'
    if index == 3: return value[::-1]
    if index == 4: return value[1:] + value[:1]
    if index == 5: return value.translate(str.maketrans('ab', 'ba'))
    raise ValueError('Unknown string operator index')


def execute(ops: tuple[int, ...], value: str) -> str:
    for op in ops:
        value = string_transform(op, value)
    return value


@dataclass(frozen=True)
class StringChallenge:
    examples: tuple[tuple[str, str], ...]
    max_depth: int
    task_id: int


class StringJudge:
    def __init__(self, target: tuple[int, ...], holdout: tuple[str, ...]):
        self.__target = target
        self.__holdout = holdout
        self.__used = False

    def final_accuracy(self, candidate: tuple[int, ...] | None) -> float:
        if self.__used:
            raise RuntimeError('Final scoring is single-use')
        self.__used = True
        if candidate is None:
            return 0.0
        return sum(execute(candidate, s) == execute(self.__target, s)
                   for s in self.__holdout) / len(self.__holdout)


def make_string_task(rng: random.Random, depth: int, task_id: int,
                     regime: str) -> tuple[StringChallenge, StringJudge]:
    if depth not in (2, 3, 4):
        raise ValueError('depth must be 2..4')
    if regime not in ('aligned', 'independent'):
        raise ValueError('Unknown regime')
    # Aligned regime has a *researcher-specified* shared index motif distribution.
    # Independent regime has no motif relationship to the byte training generator.
    if regime == 'aligned' and rng.random() < .65:
        motifs = ((4, 3), (0, 5))
        chosen: list[int] = []
        while len(chosen) < depth:
            chosen.extend(rng.choice(motifs))
        target = tuple(chosen[:depth])
    else:
        target = tuple(rng.randrange(len(STRING_OPS)) for _ in range(depth))
    # Draw disjoint observable / sealed holdout inputs without replacement.
    inputs = rng.sample(STRING_INPUTS, 32)
    examples = tuple((s, execute(target, s)) for s in inputs[:8])
    return StringChallenge(examples, depth, task_id), StringJudge(target, tuple(inputs[8:]))


@dataclass(frozen=True)
class StringSearchResult:
    program: tuple[int, ...] | None
    candidate_evaluations: int
    primitive_applications: int
    macro_uses: int


def solve_string(challenge: StringChallenge, macros: tuple[tuple[int, ...], ...] = (),
                 budget: int = 300, work_budget: int | None = None) -> StringSearchResult:
    """Bounded, non-executing-source best-first search; equal compute caps per arm.

    A primitive application is one operator over one visible example. Accounting
    for full underlying macro lengths avoids calling a macro 'free computation'.
    """
    if budget < 1 or (work_budget is not None and work_budget < 1):
        raise ValueError('Budgets must be positive')
    if not 1 <= challenge.max_depth <= 4 or not challenge.examples:
        raise ValueError('Invalid challenge')
    if len({x for x, _ in challenge.examples}) != len(challenge.examples):
        raise ValueError('Duplicate inputs')
    alphabet: tuple[tuple[int, ...], ...] = tuple((i,) for i in range(len(STRING_OPS)))
    valid_macros = tuple(dict.fromkeys(m for m in macros
                              if 2 <= len(m) <= challenge.max_depth
                              and all(type(i) is int and 0 <= i < len(STRING_OPS) for i in m)))
    alphabet += valid_macros[:128]
    # Minimum search-token cost, then primitive length. Nothing learns from tests.
    queue: list[tuple[int, int, tuple[int, ...], int]] = [(0, 0, (), 0)]
    seen: set[tuple[int, ...]] = set()
    evaluated = work = 0
    max_work = work_budget if work_budget is not None else budget * 8 * 4
    while queue and evaluated < budget:
        cost, length, seq, macro_uses = heappop(queue)
        if seq:
            if seq in seen: continue
            seen.add(seq)
            effort = length * len(challenge.examples)
            if work + effort > max_work:
                continue
            work += effort
            evaluated += 1
            if all(execute(seq, x) == y for x, y in challenge.examples):
                return StringSearchResult(seq, evaluated, work, macro_uses)
        if length >= challenge.max_depth: continue
        for item in alphabet:
            if length + len(item) > challenge.max_depth: continue
            heappush(queue, (cost + 1, length + len(item), seq + item,
                             macro_uses + (len(item) > 1)))
    return StringSearchResult(None, evaluated, work, 0)


def _summarize(records: list[dict], mode: str, challenger: str = 'baseline') -> dict:
    successes = [r[f'{mode}_pass'] for r in records]
    control = [r[f'{challenger}_pass'] for r in records]
    wins = sum(a and not b for a, b in zip(successes, control))
    losses = sum(b and not a for a, b in zip(successes, control))
    return {'pass_rate': round(mean(successes), 4),
            'delta_vs_' + challenger: round(mean(successes) - mean(control), 4),
            'only_' + mode + '_passes': wins, 'only_' + challenger + '_passes': losses,
            'candidate_evaluations': sum(r[f'{mode}_evaluations'] for r in records),
            'primitive_applications': sum(r[f'{mode}_work'] for r in records)}


def crossdomain_transfer(seed: int = 0, train_episodes: int = 100,
                         tasks_per_regime: int = 100, budget: int = 300,
                         output: Path | None = None) -> dict:
    if train_episodes < 1 or tasks_per_regime < 1 or budget < 1:
        raise ValueError('Episodes, tasks and budget must be positive')
    # The byte-domain training process never receives string-task data.
    with TemporaryDirectory() as tmp:
        training_dir = Path(tmp)
        training = run_training(train_episodes, seed, budget, training_dir)
        source = SkillLibrary(training_dir / 'skills.json').macros()
    templates = tuple(tuple(PRIMITIVES.index(op) for op in p.ops) for p in source)
    # Fixed, nonidentity shuffling independent of all target results.
    perm = list(range(len(STRING_OPS)))
    random.Random(seed ^ 0xBADC0DE).shuffle(perm)
    if perm == list(range(len(STRING_OPS))):
        perm = perm[1:] + perm[:1]
    scrambled = tuple(tuple(perm[i] for i in seq) for seq in templates)
    rng = random.Random(seed ^ 0xD0A1BEEF)
    outcomes = []
    result = {'seed': seed, 'training_episodes': train_episodes,
              'evaluation_tasks_per_regime': tasks_per_regime,
              'budget': budget, 'transferred_templates': len(templates),
              'string_primitives': list(STRING_OPS),
              'index_alignment': 'researcher-specified, not learned',
              'regimes': {}}
    for regime in ('aligned', 'independent'):
        records = []
        for task_id in range(tasks_per_regime):
            depth = rng.choice((2, 3, 4))
            task_seed = rng.randrange(2**63)
            challenge, judge_a = make_string_task(random.Random(task_seed), depth, task_id, regime)
            _, judge_b = make_string_task(random.Random(task_seed), depth, task_id, regime)
            _, judge_c = make_string_task(random.Random(task_seed), depth, task_id, regime)
            kwargs = dict(budget=budget, work_budget=budget * 8 * 4)
            adaptive = solve_string(challenge, templates, **kwargs)
            baseline = solve_string(challenge, (), **kwargs)
            negative = solve_string(challenge, scrambled, **kwargs)
            rec = {'regime': regime, 'task_id': task_id, 'depth': depth}
            for name, search, judge in (('adaptive', adaptive, judge_a),
                                        ('baseline', baseline, judge_b),
                                        ('shuffled', negative, judge_c)):
                rec[name + '_pass'] = judge.final_accuracy(search.program) == 1
                rec[name + '_evaluations'] = search.candidate_evaluations
                rec[name + '_work'] = search.primitive_applications
            records.append(rec)
            outcomes.append(rec)
        result['regimes'][regime] = {
            'tasks': tasks_per_regime,
            'baseline_pass_rate': round(mean(r['baseline_pass'] for r in records), 4),
            'baseline_candidate_evaluations': sum(r['baseline_evaluations'] for r in records),
            'baseline_primitive_applications': sum(r['baseline_work'] for r in records),
            'adaptive': _summarize(records, 'adaptive'),
            'shuffled': _summarize(records, 'shuffled')}
    result['caveat'] = ('Both domains and alignment are hand-designed. Aligned regime explicitly shares '
                        'program-index motifs; independent regime does not. This tests structural reuse '
                        'and a negative control, not learned semantic alignment or open-ended intelligence.')
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'trials.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in outcomes), encoding='utf-8')
        (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def multi_seed_crossdomain(start_seed: int = 0, runs: int = 5,
                           train_episodes: int = 100, tasks_per_regime: int = 100,
                           budget: int = 300, output: Path | None = None) -> dict:
    if not 2 <= runs <= 30:
        raise ValueError('runs must be in 2..30')
    results = [crossdomain_transfer(start_seed + i, train_episodes, tasks_per_regime, budget,
                                    output / f'seed_{start_seed + i}' if output else None)
               for i in range(runs)]
    combined: dict = {'runs': runs, 'start_seed': start_seed,
                      'train_episodes_per_seed': train_episodes,
                      'tasks_per_regime_per_seed': tasks_per_regime,
                      'budget': budget, 'regimes': {}, 'per_seed': results}
    for regime in ('aligned', 'independent'):
        metrics = [r['regimes'][regime] for r in results]
        a = [m['adaptive']['delta_vs_baseline'] for m in metrics]
        s = [m['shuffled']['delta_vs_baseline'] for m in metrics]
        combined['regimes'][regime] = {
            'mean_transfer_delta': round(mean(a), 4),
            'sd_transfer_delta': round(stdev(a), 4),
            'mean_shuffled_delta': round(mean(s), 4),
            'sd_shuffled_delta': round(stdev(s), 4),
            'per_seed_transfer_delta': a,
            'per_seed_shuffled_delta': s}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'multi_seed_summary.json').write_text(json.dumps(combined, indent=2) + '\n', encoding='utf-8')
    return combined
