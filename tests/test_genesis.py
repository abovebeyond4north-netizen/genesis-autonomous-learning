import random
import tempfile
import unittest
from pathlib import Path
from genesis.program import Program, PRIMITIVES
from genesis.tasks import make_task, Challenge
from genesis.synth import synthesize
from genesis.curriculum import Curriculum
from genesis.skills import SkillLibrary
from genesis.archive import BehaviorArchive
from genesis.trainer import run_training
from genesis.benchmark import paired_transfer, multi_seed_transfer
from genesis.world import EmpiricalPlanner, LinearWorld, world_demo
from genesis.rl import HiddenDirectionWorld, learn, QLearner


class GenesisTests(unittest.TestCase):
    def test_total_transformations(self):
        for op in PRIMITIVES:
            p = Program((op,))
            self.assertTrue(all(0 <= p(x) < 256 for x in range(256)))

    def test_invalid_program_does_not_execute(self):
        with self.assertRaises(ValueError): Program(("__import__",))

    def test_synthesis_exact(self):
        target = Program(("rotl", "xor1"))
        challenge = Challenge(tuple((x, target(x)) for x in (0, 2, 6, 13, 101, 255)), 2, 1)
        result = synthesize(challenge, budget=100)
        self.assertIsNotNone(result.program)
        self.assertTrue(all(result.program(x) == target(x) for x in range(256)))

    def test_holdouts_not_exposed_and_single_use(self):
        c, judge = make_task(random.Random(1), 2, 0)
        self.assertEqual(len(c.examples), 8)
        self.assertTrue(0 <= judge.final_accuracy(None) <= 1)
        with self.assertRaises(RuntimeError): judge.final_accuracy(None)

    def test_curriculum_explores(self):
        cur = Curriculum(random.Random(0))
        for _ in range(12):
            d = cur.pick()
            cur.observe(d, True)
        self.assertTrue(all(v["attempts"] > 0 for v in cur.stats().values()))

    def test_skill_storage_rejects_code_and_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "skills.json"
            lib = SkillLibrary(path)
            self.assertTrue(lib.add(Program(("inc", "rotl"))))
            self.assertFalse(lib.add(Program(("inc", "rotl"))))
            self.assertEqual(lib.macros(), SkillLibrary(path).macros())
            path.write_text('[["import os", "inc"]]')
            with self.assertRaises(ValueError): SkillLibrary(path)

    def test_archive_elitism(self):
        arc = BehaviorArchive()
        p = Program(("inc",))
        self.assertTrue(arc.insert(p, 1))
        self.assertFalse(arc.insert(p, 0.9))

    def test_training_reproducible(self):
        a = run_training(25, 1, 100)
        b = run_training(25, 1, 100)
        self.assertEqual(a, b)
        self.assertGreaterEqual(a["final_pass"], 1)

    def test_paired_evaluation_reproducible(self):
        a = paired_transfer(5, 20, 15, 100)
        b = paired_transfer(5, 20, 15, 100)
        self.assertEqual(a, b)
        self.assertLessEqual(a["adaptive_only_passes"], 15)
        self.assertGreater(a["trained_skills"], 0)

    def test_tabular_rl_learns_from_zero(self):
        agent = QLearner(2, seed=10)
        self.assertEqual(len(agent.q), 0)
        _, result = learn(HiddenDirectionWorld(size=4, swap=True), 500, seed=10, max_steps=30)
        self.assertGreater(result["last_block_mean_return"], 0.7)
        self.assertLess(result["last_block_mean_steps"], result["first_block_mean_steps"])

    def test_multiseed_outputs_both_variants(self):
        result = multi_seed_transfer(0, 2, 5, 5, 60)
        self.assertEqual(result["runs"], 2)
        self.assertEqual(len(result["per_seed"]), 2)

    def test_world_learning(self):
        w = LinearWorld(5, swap=True)
        self.assertEqual(w.step(0)[0], 1)
        p = EmpiricalPlanner(1)
        p.observe(0, 0, 1)
        self.assertEqual(p.action(0, 1), 0)
        result = world_demo(1, 20)
        self.assertEqual(len(result["episode_steps"]), 20)


if __name__ == "__main__": unittest.main()
