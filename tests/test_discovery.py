import json
import random
import tempfile
import unittest
from pathlib import Path

from genesis.discovery import (OpaqueHexWorld, infer_mapping, execute, make_task, solve,
                               discovery_transfer, multiseed_discovery)
from genesis.crossdomain import string_transform, STRING_OPS
from genesis.program import PRIMITIVES, transform


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.world = OpaqueHexWorld((4, 2, 5, 1, 0, 3))
        self.cal = (0, 1, 3, 17, 85, 127, 254, 255)
        self.conf = (2, 4, 5, 21, 42, 64, 128, 192)

    def test_mapping_is_inferred_from_unlabeled_observations(self):
        r = infer_mapping(self.world.apply, self.world.size, self.cal, self.conf)
        self.assertFalse(r['abstained'])
        self.assertEqual(r['mapping'], [self.world.target_index(i) for i in range(6)])
        self.assertEqual(r['oracle_queries'], 96)
        self.assertTrue(all(execute(self.world, (r['mapping'][i],), '0102ff') ==
                            bytes(transform(PRIMITIVES[i], x) for x in (1, 2, 255)).hex()
                            for i in range(6)))

    def test_unrelated_string_operations_causes_abstention(self):
        r = infer_mapping(string_transform, len(STRING_OPS), self.cal, self.conf)
        self.assertTrue(r['abstained'])
        self.assertIsNone(r['mapping'])

    def test_tamper_confirmation_causes_abstention(self):
        def tampered(op, text):
            if text == f'{self.conf[0]:02x}':
                return 'ffff'
            return self.world.apply(op, text)
        r = infer_mapping(tampered, 6, self.cal, self.conf)
        self.assertTrue(r['abstained'])
        self.assertTrue(r['rejected_confirmation'])

    def test_reject_ambiguous_queries_and_invalid_anchor_splits(self):
        r = infer_mapping(lambda op, x: x, 6, self.cal, self.conf)
        self.assertTrue(r['abstained'])
        self.assertTrue(r['rejected_ambiguous_or_incompatible'])
        with self.assertRaises(ValueError):
            infer_mapping(self.world.apply, 6, (1, 2), (2, 3))
        with self.assertRaises(ValueError):
            infer_mapping(self.world.apply, 6, (1, 1), (2, 3))
        with self.assertRaises(ValueError):
            infer_mapping(self.world.apply, 6, (-1,), (5,))

    def test_world_rejects_bad_codes_and_programs(self):
        for code in ('', 'a', '0G', '11 ', '0a' * 9):
            with self.assertRaises(ValueError):
                self.world.apply(0, code)
        for index in (-1, True, 6):
            with self.assertRaises(ValueError):
                self.world.apply(index, '0001')

    def test_disjoint_task_observations_and_sealed_judging(self):
        c, judge = make_task(self.world, random.Random(42), 4, 0, 'related')
        self.assertEqual(len(c.examples), 8)
        self.assertTrue(all(len(x) >= 4 for x, _ in c.examples))
        self.assertEqual(len(set(x for x, _ in c.examples)), 8)
        self.assertLessEqual(judge.final_accuracy((0, 1)), 1.0)
        with self.assertRaises(RuntimeError):
            judge.final_accuracy((0, 1))

    def test_bounded_search(self):
        c, _ = make_task(self.world, random.Random(2), 3, 0, 'independent')
        for macros in ((), ((0, 1), (2, 3)), ((-1, 2), (True, 2))):
            r = solve(self.world, c, macros, budget=50, work_budget=400)
            self.assertLessEqual(r.evaluations, 50)
            self.assertLessEqual(r.primitive_work, 400)
        with self.assertRaises(ValueError):
            solve(self.world, c, budget=0)

    def test_experiment_reproducible_and_negative_control(self):
        r = discovery_transfer(seed=2, train_episodes=12, tasks_per_regime=5, budget=80)
        self.assertEqual(r, discovery_transfer(seed=2, train_episodes=12, tasks_per_regime=5, budget=80))
        self.assertTrue(r['mapping_correct_after_sealed_teacher_check'])
        self.assertTrue(r['unrelated_string_mapping_abstained'])
        self.assertEqual(set(r['regimes']), {'related', 'independent'})
        for arm in ('baseline', 'learned', 'shuffled'):
            self.assertIn(arm, r['regimes']['related'])

    def test_multiseed_writes_reproducible_records(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            r = multiseed_discovery(0, 2, 8, 5, 90, p)
            self.assertEqual(len(r['per_seed']), 2)
            self.assertTrue((p / 'multi_seed_summary.json').exists())
            self.assertEqual(len((p / 'seed_0' / 'trials.jsonl').read_text().splitlines()), 10)
            self.assertEqual(json.loads((p / 'multi_seed_summary.json').read_text())['runs'], 2)


if __name__ == '__main__':
    unittest.main()
