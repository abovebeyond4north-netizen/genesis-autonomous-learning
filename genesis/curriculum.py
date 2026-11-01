"""Online absolute-learning-progress scheduler with a UCB exploration term."""
from __future__ import annotations
from collections import deque
import math
import random


class Curriculum:
    def __init__(self, rng: random.Random, depths: tuple[int, ...] = (1, 2, 3, 4)):
        if not depths or min(depths) < 1: raise ValueError("Invalid depths")
        self.rng, self.depths = rng, depths
        self._history: dict[int, deque[float]] = {d: deque(maxlen=16) for d in depths}
        self._counts = {d: 0 for d in depths}

    def pick(self) -> int:
        untried = [d for d in self.depths if not self._counts[d]]
        if untried: return self.rng.choice(untried)
        total = sum(self._counts.values())
        def score(d: int) -> float:
            h = list(self._history[d])
            if len(h) < 4: progress = 1.0
            else:
                mid = len(h) // 2
                progress = abs(sum(h[mid:]) / len(h[mid:]) - sum(h[:mid]) / mid)
            return progress + 0.25 * math.sqrt(math.log(total + 1) / self._counts[d])
        top = max(score(d) for d in self.depths)
        return self.rng.choice([d for d in self.depths if abs(score(d) - top) < 1e-12])

    def observe(self, depth: int, success: bool) -> None:
        self._counts[depth] += 1
        self._history[depth].append(float(success))

    def stats(self) -> dict[str, dict[str, float | int]]:
        return {str(d): {"attempts": self._counts[d], "recent_success": round(sum(self._history[d]) / max(1, len(self._history[d])), 3)} for d in self.depths}
