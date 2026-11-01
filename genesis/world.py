"""Empirical transition learning and planning in a novel finite-state environment."""
from __future__ import annotations
from collections import deque, defaultdict
from random import Random


class LinearWorld:
    def __init__(self, size: int = 7, swap: bool = False):
        if size < 3: raise ValueError("World too small")
        self.size, self.swap, self.position = size, swap, 0

    def reset(self, start: int = 0) -> int:
        if not 0 <= start < self.size: raise ValueError("Invalid start")
        self.position = start
        return self.position

    def step(self, action: int) -> tuple[int, float]:
        if action not in (0, 1): raise ValueError("Invalid action")
        direction = (-1 if action == 0 else 1) * (-1 if self.swap else 1)
        self.position = max(0, min(self.size - 1, self.position + direction))
        return self.position, float(self.position == self.size - 1)


class EmpiricalPlanner:
    def __init__(self, seed: int = 0):
        self.rng = Random(seed)
        self.counts: dict[tuple[int, int], int] = defaultdict(int)
        self.next_states: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def observe(self, state: int, action: int, next_state: int) -> None:
        self.counts[state, action] += 1
        self.next_states[state, action][next_state] += 1

    def action(self, state: int, goal: int) -> int:
        # BFS over only observed transition edges; unseen edges are explored.
        queue = deque([(state, None)])
        seen = {state}
        while queue:
            node, first = queue.popleft()
            if node == goal and first is not None: return first
            for action in (0, 1):
                successors = self.next_states.get((node, action), {})
                if not successors: continue
                nxt = max(successors, key=successors.get)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, action if first is None else first))
        # Select least-explored action, randomly breaking ties.
        choices = [0, 1]
        self.rng.shuffle(choices)
        return min(choices, key=lambda a: self.counts[state, a])


def world_demo(seed: int = 0, episodes: int = 30, size: int = 7) -> dict:
    world = LinearWorld(size, swap=bool(seed % 2))
    learner = EmpiricalPlanner(seed)
    steps = []
    for _ in range(episodes):
        s = world.reset()
        for t in range(size * 8):
            a = learner.action(s, size - 1)
            nxt, reward = world.step(a)
            learner.observe(s, a, nxt)
            s = nxt
            if reward:
                steps.append(t + 1)
                break
        else: steps.append(size * 8)
    return {"episode_steps": steps, "first_five_mean": sum(steps[:5]) / min(5, len(steps)),
            "last_five_mean": sum(steps[-5:]) / min(5, len(steps))}
