"""Procedural training tasks with a deliberately sealed evaluator interface."""
from __future__ import annotations
from dataclasses import dataclass
import random
from .program import Program, PRIMITIVES


@dataclass(frozen=True)
class Challenge:
    examples: tuple[tuple[int, int], ...]
    max_depth: int
    task_id: int


class Judge:
    """Holdout samples and target stay outside the synthesizer interface."""
    def __init__(self, target: Program, validation: tuple[int, ...], final: tuple[int, ...]):
        self._target, self._validation, self._final = target, validation, final
        self._validation_used = False
        self._final_used = False

    def validation_accuracy(self, program: Program | None) -> float:
        if self._validation_used:
            raise RuntimeError("Validation is single-use")
        self._validation_used = True
        if program is None: return 0.0
        return sum(program(x) == self._target(x) for x in self._validation) / len(self._validation)

    def final_accuracy(self, program: Program | None) -> float:
        if self._final_used:
            raise RuntimeError("Final evaluation is single-use")
        self._final_used = True
        if program is None: return 0.0
        return sum(program(x) == self._target(x) for x in self._final) / len(self._final)


def make_task(rng: random.Random, depth: int, task_id: int, motifs: tuple[tuple[str, ...], ...] = ()) -> tuple[Challenge, Judge]:
    if depth < 1 or depth > 8:
        raise ValueError("depth must be in 1..8")
    if motifs and rng.random() < 0.65:
        ops: list[str] = []
        while len(ops) < depth:
            ops.extend(rng.choice(motifs))
        chosen = tuple(ops[:depth])
    else:
        chosen = tuple(rng.choice(PRIMITIVES) for _ in range(depth))
    target = Program(chosen)
    values = rng.sample(range(256), 48)
    examples = tuple((x, target(x)) for x in values[:8])
    return Challenge(examples, depth, task_id), Judge(target, tuple(values[8:24]), tuple(values[24:]))
