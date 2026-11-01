"""Data-only persistent skills, with schema validation and bounded growth."""
from __future__ import annotations
import json
from pathlib import Path
from .program import Program


class SkillLibrary:
    def __init__(self, path: Path | None = None, limit: int = 128):
        self.path, self.limit = path, limit
        self._items: dict[str, Program] = {}
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or len(data) > limit:
                raise ValueError("Invalid skill database")
            for item in data:
                if not isinstance(item, list) or len(item) < 2 or len(item) > 8:
                    raise ValueError("Malformed skill")
                program = Program(tuple(item))
                self._items[program.identifier] = program

    def add(self, program: Program) -> bool:
        if not 2 <= len(program.ops) <= 8: return False
        if program.identifier in self._items: return False
        if len(self._items) >= self.limit: return False
        self._items[program.identifier] = program
        self.save()
        return True

    def save(self) -> None:
        if self.path is None: return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps([list(p.ops) for p in self._items.values()], indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def macros(self) -> tuple[Program, ...]:
        return tuple(self._items.values())

    def __len__(self) -> int:
        return len(self._items)
