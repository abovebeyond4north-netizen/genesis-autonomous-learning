"""Budget-limited observational program synthesizer with optional macro reuse."""
from __future__ import annotations
from dataclasses import dataclass
from heapq import heappush, heappop
from .program import Program, PRIMITIVES
from .tasks import Challenge


@dataclass(frozen=True)
class SearchResult:
    program: Program | None
    evaluated: int
    expanded: int
    macro_uses: int


def synthesize(challenge: Challenge, macros: tuple[Program, ...] = (), budget: int = 512) -> SearchResult:
    if budget < 1: raise ValueError("budget must be positive")
    # Search cost counts reusable verified macros as one learned operation.
    # Priorities: number of search operators, underlying program length, stable lexical order.
    alphabet = tuple((p,) for p in PRIMITIVES) + tuple(p.ops for p in macros)
    heap: list[tuple[int, int, tuple[str, ...], int, int]] = []
    counter = 0
    heappush(heap, (0, 0, (), counter, 0))
    seen_programs: set[tuple[str, ...]] = set()
    signatures: set[tuple[int, ...]] = set()
    evaluated, expanded = 0, 0
    while heap and evaluated < budget:
        search_cost, length, seq, _, macro_uses = heappop(heap)
        if seq:
            if seq in seen_programs: continue
            seen_programs.add(seq)
            prog = Program(seq)
            signature = tuple(prog(x) for x, _ in challenge.examples)
            evaluated += 1
            # Deduplicate equivalent representations on the observed inputs.
            if signature in signatures: continue
            signatures.add(signature)
            if all(prog(x) == y for (x, y) in challenge.examples):
                return SearchResult(prog, evaluated, expanded, macro_uses)
        if length >= challenge.max_depth: continue
        expanded += 1
        for item in alphabet:
            new_length = length + len(item)
            if new_length > challenge.max_depth: continue
            counter += 1
            heappush(heap, (search_cost + 1, new_length, seq + item, counter, macro_uses + (len(item) > 1)))
    return SearchResult(None, evaluated, expanded, 0)
