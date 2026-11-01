"""Lightweight MAP-Elites-inspired behavior archive (not a pyribs implementation)."""
from __future__ import annotations
from dataclasses import dataclass
from .program import Program


@dataclass(frozen=True)
class Elite:
    program: Program
    score: float


class BehaviorArchive:
    def __init__(self):
        self.cells: dict[tuple[int, ...], Elite] = {}

    @staticmethod
    def descriptor(program: Program) -> tuple[int, ...]:
        # Fixed canonical probes; coarse bins encourage functional diversity.
        return tuple(program(x) // 32 for x in (0, 3, 37, 119, 251))

    def insert(self, program: Program, score: float) -> bool:
        key = self.descriptor(program)
        old = self.cells.get(key)
        if old is not None and (score, -len(program.ops)) <= (old.score, -len(old.program.ops)):
            return False
        self.cells[key] = Elite(program, score)
        return True

    def __len__(self) -> int: return len(self.cells)
