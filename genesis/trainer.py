"""From-scratch learner: procedural tasks, adaptive curriculum, synthesis, holdout promotion."""
from __future__ import annotations
import json
from pathlib import Path
import random
from .archive import BehaviorArchive
from .curriculum import Curriculum
from .skills import SkillLibrary
from .synth import synthesize
from .tasks import make_task


def run_training(episodes: int = 100, seed: int = 0, budget: int = 300, output: Path | None = None,
                 use_macros: bool = True) -> dict:
    if episodes < 1 or budget < 1: raise ValueError("Episodes and budget must be positive")
    rng = random.Random(seed)
    curriculum = Curriculum(rng)
    library = SkillLibrary(output / "skills.json" if output and use_macros else None)
    archive = BehaviorArchive()
    events = []
    total_eval, validated, final_pass = 0, 0, 0
    for episode in range(episodes):
        depth = curriculum.pick()
        challenge, judge = make_task(rng, depth, episode, motifs=(("rotl", "xor1"), ("inc", "invert")))
        result = synthesize(challenge, library.macros() if use_macros else (), budget)
        val_acc = judge.validation_accuracy(result.program)
        final_acc = judge.final_accuracy(result.program)  # report-only: never used to select a candidate
        success = val_acc == 1.0
        curriculum.observe(depth, success)
        if success and result.program is not None:
            validated += 1
            archive.insert(result.program, val_acc)
            if use_macros: library.add(result.program)
        final_pass += final_acc == 1.0
        total_eval += result.evaluated
        events.append({"episode": episode, "depth": depth, "solved_validation": success,
                       "final_accuracy": final_acc, "evaluations": result.evaluated,
                       "macro_uses": result.macro_uses,
                       "program": list(result.program.ops) if result.program else None})
    summary = {"episodes": episodes, "seed": seed, "budget_per_task": budget,
               "validation_pass": validated, "final_pass": final_pass,
               "validation_rate": round(validated / episodes, 4),
               "final_rate": round(final_pass / episodes, 4),
               "total_candidate_evaluations": total_eval, "skills": len(library) if use_macros else 0,
               "behavior_cells": len(archive), "curriculum": curriculum.stats()}
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
