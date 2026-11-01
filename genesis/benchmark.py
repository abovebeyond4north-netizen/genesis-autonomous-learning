"""Strictly paired, frozen transfer evaluation after learning reusable skills."""
from __future__ import annotations
from pathlib import Path
import random
from .tasks import make_task
from .skills import SkillLibrary
from .synth import synthesize
from .trainer import run_training


def paired_transfer(seed: int = 42, train_episodes: int = 100, eval_tasks: int = 100,
                    budget: int = 300, output: Path | None = None) -> dict:
    if eval_tasks < 1: raise ValueError("eval_tasks must be positive")
    # Training and evaluation use independent PRNG streams. No test feedback changes macros.
    library_path = output / "adaptive_train" if output is not None else None
    if library_path is None:
        # Temporary storage only for benchmark's learner; no external side effects.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            return paired_transfer(seed, train_episodes, eval_tasks, budget, Path(d))
    training = run_training(train_episodes, seed, budget, library_path, use_macros=True)
    macros = SkillLibrary(library_path / "skills.json").macros()
    rng = random.Random(seed ^ 0xA5A5F00D)
    records = []
    for i in range(eval_tasks):
        task_seed = rng.randrange(2**63)
        depth = rng.choice((2, 3, 4))
        a, j1 = make_task(random.Random(task_seed), depth, i,
                          motifs=(("rotl", "xor1"), ("inc", "invert")))
        b, j2 = make_task(random.Random(task_seed), depth, i,
                          motifs=(("rotl", "xor1"), ("inc", "invert")))
        assert a == b
        adapted = synthesize(a, macros, budget)
        baseline = synthesize(b, (), budget)
        # Only the final evaluator scores these models; sealed input samples were never read by search.
        acc_a = j1.final_accuracy(adapted.program)
        acc_b = j2.final_accuracy(baseline.program)
        records.append({"task": i, "depth": depth, "adaptive_pass": acc_a == 1,
                        "baseline_pass": acc_b == 1,
                        "adaptive_accuracy": acc_a, "baseline_accuracy": acc_b,
                        "adaptive_evaluations": adapted.evaluated, "baseline_evaluations": baseline.evaluated})
    a_pass = sum(r["adaptive_pass"] for r in records)
    b_pass = sum(r["baseline_pass"] for r in records)
    a_eval = sum(r["adaptive_evaluations"] for r in records)
    b_eval = sum(r["baseline_evaluations"] for r in records)
    paired_wins = sum(r["adaptive_pass"] and not r["baseline_pass"] for r in records)
    paired_losses = sum(r["baseline_pass"] and not r["adaptive_pass"] for r in records)
    summary = {"seed": seed, "train_episodes": train_episodes, "evaluation_tasks": eval_tasks,
               "budget_per_task": budget, "trained_skills": len(macros),
               "adaptive_pass": a_pass, "baseline_pass": b_pass,
               "adaptive_pass_rate": round(a_pass / eval_tasks, 4),
               "baseline_pass_rate": round(b_pass / eval_tasks, 4),
               "delta_pass_rate": round((a_pass - b_pass) / eval_tasks, 4),
               "adaptive_candidate_evaluations": a_eval,
               "baseline_candidate_evaluations": b_eval,
               "adaptive_only_passes": paired_wins, "baseline_only_passes": paired_losses,
               "training": training,
               "notes": "Frozen macros, fixed heldout tasks, equal candidate budgets; synthetic DSL only. A single seed is not a generalization confidence interval."}
    if output is not None:
        import json
        output.mkdir(parents=True, exist_ok=True)
        (output / "paired_trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        (output / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def multi_seed_transfer(start_seed: int = 0, runs: int = 5, train_episodes: int = 150,
                        eval_tasks: int = 100, budget: int = 300,
                        output: Path | None = None) -> dict:
    """Independent replications with unpooled per-seed outcomes."""
    from statistics import mean, stdev
    if runs < 2 or runs > 50: raise ValueError("runs must be in 2..50")
    trials = [paired_transfer(start_seed + i, train_episodes, eval_tasks, budget,
                              output / f"seed_{start_seed + i}" if output else None)
              for i in range(runs)]
    diffs = [t["delta_pass_rate"] for t in trials]
    savings = [1 - t["adaptive_candidate_evaluations"] / max(1, t["baseline_candidate_evaluations"])
               for t in trials]
    result = {"runs": runs, "train_episodes_per_seed": train_episodes,
              "eval_tasks_per_seed": eval_tasks, "budget_per_task": budget,
              "mean_pass_rate_delta": round(mean(diffs), 4),
              "stddev_pass_rate_delta": round(stdev(diffs), 4),
              "mean_relative_evaluation_saving": round(mean(savings), 4),
              "per_seed": [{"seed": t["seed"], "delta_pass_rate": t["delta_pass_rate"],
                            "adaptive_pass": t["adaptive_pass"], "baseline_pass": t["baseline_pass"],
                            "adaptive_only_passes": t["adaptive_only_passes"],
                            "baseline_only_passes": t["baseline_only_passes"],
                            "adaptive_candidate_evaluations": t["adaptive_candidate_evaluations"],
                            "baseline_candidate_evaluations": t["baseline_candidate_evaluations"]}
                           for t in trials],
              "caveat": "Repeated paired synthetic DSL benchmarks; performance on new domains and true open-endedness remain untested."}
    if output:
        import json
        output.mkdir(parents=True, exist_ok=True)
        (output / "multi_seed_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
