"""Run: python -m genesis train|bench|world"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .trainer import run_training
from .benchmark import paired_transfer, multi_seed_transfer
from .crossdomain import crossdomain_transfer, multi_seed_crossdomain
from .discovery import discovery_transfer, multiseed_discovery
from .emergence import single_trial as emergence_trial, multiseed as emergence_multiseed, representation_snapshot
from .partial import trial as partial_trial, multiseed as partial_multiseed
from .world import world_demo
from .rl import HiddenDirectionWorld, learn, learn_gymnasium


def main() -> None:
    p = argparse.ArgumentParser(description="Genesis: from-scratch autonomous learning experiments")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "bench"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--episodes", type=int, default=100)
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--budget", type=int, default=300)
        cmd.add_argument("--output", type=Path, default=None)
    pair = sub.add_parser("paired")
    pair.add_argument("--episodes", type=int, default=100, help="Training episodes")
    pair.add_argument("--tasks", type=int, default=100, help="Frozen heldout evaluation tasks")
    pair.add_argument("--seed", type=int, default=42)
    pair.add_argument("--budget", type=int, default=300)
    pair.add_argument("--output", type=Path, default=None)
    multi = sub.add_parser("multiseed")
    multi.add_argument("--runs", type=int, default=5)
    multi.add_argument("--episodes", type=int, default=150)
    multi.add_argument("--tasks", type=int, default=100)
    multi.add_argument("--seed", type=int, default=0)
    multi.add_argument("--budget", type=int, default=300)
    multi.add_argument("--output", type=Path, default=None)
    cross = sub.add_parser("crossdomain")
    cross.add_argument("--runs", type=int, default=5)
    cross.add_argument("--episodes", type=int, default=100)
    cross.add_argument("--tasks", type=int, default=100)
    cross.add_argument("--budget", type=int, default=300)
    cross.add_argument("--seed", type=int, default=0)
    cross.add_argument("--output", type=Path, default=None)
    disc = sub.add_parser("discover")
    disc.add_argument("--runs", type=int, default=5)
    disc.add_argument("--episodes", type=int, default=100)
    disc.add_argument("--tasks", type=int, default=100)
    disc.add_argument("--budget", type=int, default=300)
    disc.add_argument("--seed", type=int, default=0)
    disc.add_argument("--output", type=Path, default=None)
    emerge = sub.add_parser("emerge", help="Learn latent predictive states from opaque transitions")
    emerge.add_argument("--runs", type=int, default=30)
    emerge.add_argument("--train-steps", type=int, default=2600)
    emerge.add_argument("--test-steps", type=int, default=1200)
    emerge.add_argument("--seed", type=int, default=0)
    emerge.add_argument("--output", type=Path, default=None)
    partial = sub.add_parser("partial", help="Infer predictive history states from opaque stochastic POMDP observations")
    partial.add_argument("--runs", type=int, default=10)
    partial.add_argument("--episodes", type=int, default=30)
    partial.add_argument("--steps", type=int, default=250)
    partial.add_argument("--max-depth", type=int, default=3)
    partial.add_argument("--seed", type=int, default=0)
    partial.add_argument("--output", type=Path, default=None)
    r = sub.add_parser("rl")
    r.add_argument("--episodes", type=int, default=500)
    r.add_argument("--seed", type=int, default=1)
    g = sub.add_parser("gym")
    g.add_argument("--env-id", type=str, default="FrozenLake-v1")
    g.add_argument("--episodes", type=int, default=2000)
    g.add_argument("--seed", type=int, default=1)
    w = sub.add_parser("world")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()
    if args.command == "train":
        result = run_training(args.episodes, args.seed, args.budget, args.output)
    elif args.command == "multiseed":
        result = multi_seed_transfer(args.seed, args.runs, args.episodes, args.tasks, args.budget, args.output)
    elif args.command == "paired":
        result = paired_transfer(args.seed, args.episodes, args.tasks, args.budget, args.output)
    elif args.command == "crossdomain":
        result = (multi_seed_crossdomain(args.seed, args.runs, args.episodes, args.tasks, args.budget, args.output)
                  if args.runs > 1 else crossdomain_transfer(args.seed, args.episodes, args.tasks, args.budget, args.output))
    elif args.command == "discover":
        result = (multiseed_discovery(args.seed, args.runs, args.episodes, args.tasks, args.budget, args.output)
                  if args.runs > 1 else discovery_transfer(args.seed, args.episodes, args.tasks, args.budget, args.output))
    elif args.command == "emerge":
        result = (emergence_multiseed(args.seed, args.runs, args.train_steps, args.test_steps)
                  if args.runs > 1 else emergence_trial(args.seed, args.train_steps, args.test_steps))
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            (args.output / "invented_representation.json").write_text(
                json.dumps(representation_snapshot(args.seed, args.train_steps), indent=2) + "\n",
                encoding="utf-8")
    elif args.command == "partial":
        result = partial_multiseed(args.seed, args.runs, args.episodes, args.steps, args.max_depth)
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            (args.output / "trials.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in result["per_seed"]), encoding="utf-8")
    elif args.command == "rl":
        _, result = learn(HiddenDirectionWorld(), args.episodes, args.seed)
    elif args.command == "gym":
        result = learn_gymnasium(args.env_id, args.episodes, args.seed)
    elif args.command == "world":
        result = world_demo(args.seed, args.episodes)
    else:
        # Matched seed, task budget, curriculum policy; progression diverges after feedback.
        adaptive = run_training(args.episodes, args.seed, args.budget, args.output / "adaptive" if args.output else None, True)
        baseline = run_training(args.episodes, args.seed, args.budget, args.output / "baseline" if args.output else None, False)
        result = {"adaptive": adaptive, "no_skill_reuse": baseline,
                  "delta_final_rate": round(adaptive["final_rate"] - baseline["final_rate"], 4),
                  "caveat": "Adaptive curricula yield different task sequences; this is an online ablation, not a paired generalization estimate."}
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
