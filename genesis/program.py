"""Finite, total 8-bit transformation language. No eval or arbitrary execution."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256

PRIMITIVES = ("inc", "dec", "double", "xor1", "rotl", "invert")


def transform(op: str, value: int) -> int:
    x = value & 255
    if op == "inc": return (x + 1) & 255
    if op == "dec": return (x - 1) & 255
    if op == "double": return (x * 2) & 255
    if op == "xor1": return x ^ 1
    if op == "rotl": return ((x << 1) | (x >> 7)) & 255
    if op == "invert": return x ^ 255
    raise ValueError(f"Unknown primitive: {op}")


@dataclass(frozen=True)
class Program:
    ops: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.ops or any(op not in PRIMITIVES for op in self.ops):
            raise ValueError("Only nonempty primitive sequences are allowed")

    def __call__(self, value: int) -> int:
        for op in self.ops:
            value = transform(op, value)
        return value

    @property
    def identifier(self) -> str:
        return "s_" + sha256(",".join(self.ops).encode()).hexdigest()[:12]

    def __str__(self) -> str:
        return " -> ".join(self.ops)
