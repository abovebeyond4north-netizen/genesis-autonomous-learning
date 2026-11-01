"""Actual tabular reinforcement learning from no pretrained weights.

For environments whose observation and action spaces are both finite and discrete.
"""
from __future__ import annotations
from collections import defaultdict
import random
from typing import Protocol


class DiscreteEnv(Protocol):
    action_count: int
    def reset(self, seed: int) -> int: ...
    def step(self, action: int) -> tuple[int, float, bool]: ...


class QLearner:
    def __init__(self, action_count: int, seed: int = 0, alpha: float = 0.3,
                 gamma: float = 0.98, epsilon: float = 0.3):
        if action_count < 2 or not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= epsilon <= 1:
            raise ValueError("Invalid learner settings")
        self.n = action_count
        self.rng = random.Random(seed)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q: dict[tuple[int, int], float] = defaultdict(float)

    def choose(self, state: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n)
        vals = [self.q[state, a] for a in range(self.n)]
        best = max(vals)
        ties = [i for i, v in enumerate(vals) if v == best]
        return self.rng.choice(ties) if explore else ties[0]

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        if action not in range(self.n): raise ValueError("Invalid action")
        estimate = reward + (0 if done else self.gamma * max(self.q[next_state, a] for a in range(self.n)))
        self.q[state, action] += self.alpha * (estimate - self.q[state, action])


def learn(env: DiscreteEnv, episodes: int = 500, seed: int = 0,
          max_steps: int = 100, epsilon_start: float = 0.5,
          epsilon_end: float = 0.05) -> tuple[QLearner, dict]:
    if episodes < 1 or max_steps < 1 or not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("Invalid training schedule")
    learner = QLearner(env.action_count, seed=seed, epsilon=epsilon_start)
    rewards = []
    lengths = []
    for ep in range(episodes):
        learner.epsilon = epsilon_start + (epsilon_end - epsilon_start) * ep / max(1, episodes - 1)
        s = env.reset(seed + ep)
        total = 0.0
        for step in range(max_steps):
            action = learner.choose(s)
            nxt, reward, done = env.step(action)
            learner.update(s, action, reward, nxt, done)
            total += reward
            s = nxt
            if done: break
        rewards.append(total)
        lengths.append(step + 1)
    block = max(1, episodes // 5)
    return learner, {"episodes": episodes, "states_learned": len({s for s, a in learner.q}),
                     "first_block_mean_return": sum(rewards[:block]) / block,
                     "last_block_mean_return": sum(rewards[-block:]) / block,
                     "first_block_mean_steps": sum(lengths[:block]) / block,
                     "last_block_mean_steps": sum(lengths[-block:]) / block}


class HiddenDirectionWorld:
    """Task with unknown action semantics; only integers and rewards are exposed."""
    action_count = 2
    def __init__(self, size: int = 7, swap: bool = True):
        from .world import LinearWorld
        self.world = LinearWorld(size, swap)

    def reset(self, seed: int) -> int:
        return self.world.reset(0)

    def step(self, action: int) -> tuple[int, float, bool]:
        state, reward = self.world.step(action)
        return state, reward, reward > 0


def learn_gymnasium(env_id: str = "FrozenLake-v1", episodes: int = 2000,
                    seed: int = 0, max_steps: int = 100) -> dict:
    """Optional real Gymnasium connector, no fixed network or pretrained weights."""
    try:
        import gymnasium as gym
        from gymnasium.spaces import Discrete
    except ImportError as exc:
        raise RuntimeError("Run `pip install gymnasium` to enable this adapter") from exc
    env = gym.make(env_id)
    try:
        if not isinstance(env.observation_space, Discrete) or not isinstance(env.action_space, Discrete):
            raise ValueError("Adapter supports only discrete scalar observations and actions")

        class Adapter:
            action_count = int(env.action_space.n)
            def reset(self, seed: int) -> int:
                s, _ = env.reset(seed=seed)
                return int(s)
            def step(self, action: int) -> tuple[int, float, bool]:
                s, reward, terminated, truncated, _ = env.step(action)
                return int(s), float(reward), bool(terminated or truncated)

        learner, stats = learn(Adapter(), episodes, seed, max_steps)
        return {"environment": env_id, **stats}
    finally:
        env.close()
