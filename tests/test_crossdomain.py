import random
import tempfile
import unittest
from pathlib import Path
from genesis.crossdomain import (STRING_OPS, string_transform, execute, make_string_task,
                                 solve_string, crossdomain_transfer, multi_seed_crossdomain)


class CrossDomainTests(unittest.TestCase):
    def test_total_operator_language_and_invalid_index(self):
        for i in range(len(STRING_OPS)):
            for s in ('', 'a', 'abba', 'bbca'):
                self.assertIsInstance(string_transform(i, s), str)
        with self.assertRaises(ValueError): string_transform(100, 'a')

    def test_independent_inputs_and_final_single_use(self):
        c, judge = make_string_task(random.Random(44), 3, 7, 'independent')
        self.assertEqual(len(c.examples), 8)
        self.assertEqual(len(set(x for x, _ in c.examples)), 8)
        self.assertTrue(0 <= judge.final_accuracy((0, 5)) <= 1)
        with self.assertRaises(RuntimeError): judge.final_accuracy((0, 5))

    def test_finds_longer_compositions_without_macro(self):
        target = (3, 4, 5)
        examples = tuple((x, execute(target, x)) for x in ('abc', 'bca', 'abb', 'cab'))
        from genesis.crossdomain import StringChallenge
        solved = solve_string(StringChallenge(examples, 3, 0), budget=300)
        self.assertIsNotNone(solved.program)
        self.assertTrue(all(execute(solved.program, x) == execute(target, x)
                            for x in ('abc', 'bca', 'abb', 'cab')))

    def test_equal_work_cap_and_nonexecuting_input(self):
        from genesis.crossdomain import StringChallenge
        c = StringChallenge((('abc', 'other'),), 4, 0)
        for macro in ((), ((1, 2, 3, 4),), ((-1, 5),), ((3, 4),)):
            r = solve_string(c, macro, 30, 40)
            self.assertLessEqual(r.candidate_evaluations, 30)
            self.assertLessEqual(r.primitive_applications, 40)
        with self.assertRaises(ValueError): solve_string(c, budget=0)

    def test_reproducible_two_regimes_three_arms(self):
        one = crossdomain_transfer(seed=3, train_episodes=12, tasks_per_regime=8, budget=70)
        two = crossdomain_transfer(seed=3, train_episodes=12, tasks_per_regime=8, budget=70)
        self.assertEqual(one, two)
        self.assertSetEqual(set(one['regimes']), {'aligned', 'independent'})
        for mode in one['regimes'].values():
            self.assertIn('adaptive', mode)
            self.assertIn('shuffled', mode)
            self.assertLessEqual(mode['adaptive']['pass_rate'], 1)

    def test_multi_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as d:
            r = multi_seed_crossdomain(1, 2, 6, 6, 60, Path(d))
            self.assertEqual(r['runs'], 2)
            self.assertTrue((Path(d) / 'multi_seed_summary.json').exists())

if __name__ == '__main__': unittest.main()
