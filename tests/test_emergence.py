"""Regression and falsification tests for induced opaque predictive states."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from random import Random
from genesis.emergence import (
    InsufficientEvidence, Representation, alignment_consistency,
    discover_alignment, induce, make_worlds, multiseed, observe,
    partition_f1, prediction_loss, representation_snapshot,
    shuffled_representation, single_trial,
)


class EmergenceTests(unittest.TestCase):
    def test_disjoint_observation_and_action_codebooks(self):
        a, b, c = make_worlds(42)
        tokens = []
        for w in (a, b, c):
            tokens += list(w.actions)
            tokens += [x for g in w.aliases for x in g]
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertTrue(all(15 <= len(x) <= 23 for x in tokens))

    def test_induced_states_not_supplied(self):
        a, b, _ = make_worlds(3)
        left = induce(observe(a, 2600, 55321))
        right = induce(observe(b, 2600, 55322))
        self.assertEqual(left.symbol_count, 21)
        self.assertEqual(len(left.classes), 7)
        self.assertEqual(partition_f1(left, a), 1.0)
        self.assertEqual(partition_f1(right, b), 1.0)

    def test_opaque_graph_match_and_unrelated_rejection(self):
        a, b, c = make_worlds(10)
        left = induce(observe(a, 2600, 1234))
        right = induce(observe(b, 2600, 1235))
        unrelated = induce(observe(c, 2600, 1236))
        m = discover_alignment(left, right)
        self.assertIsNotNone(m)
        self.assertEqual(alignment_consistency(left, right, m), 1)
        self.assertIsNone(discover_alignment(left, unrelated))

    def test_heldout_nll_vs_controls(self):
        _, target, _ = make_worlds(5)
        train, test = observe(target, 2600, 44), observe(target, 1200, 45)
        rep = induce(train)
        learned = prediction_loss(train, test, rep)
        self.assertLess(learned, prediction_loss(train, test, None))
        self.assertLess(learned, prediction_loss(train, test,
                                                 shuffled_representation(rep, 21)))

    def test_low_data_abstention(self):
        a, _, _ = make_worlds(7)
        with self.assertRaises(InsufficientEvidence):
            induce(observe(a, 200, 5))
        with self.assertRaises(InsufficientEvidence):
            induce([])

    def test_reject_ambiguous_nondeterministic_quotient(self):
        sample = []
        for x in ("a", "b"):
            for y in ("c", "d"):
                sample += [(x, "go", y)] * 20
        sample += [("c", "go", "a")] * 40
        sample += [("d", "go", "b")] * 40
        with self.assertRaises(InsufficientEvidence):
            induce(sample)

    def test_representation_is_invariant_to_token_renaming(self):
        a, _, _ = make_worlds(17)
        samples = observe(a, 2600, 888)
        rng = Random(290)
        values = sorted({x for x, _, y in samples for x in (x, y)})
        actions = sorted({act for _, act, _ in samples})
        names = [f"opaque{n:04d}" for n in range(len(values))]
        motors = [f"motor{n:04d}" for n in range(len(actions))]
        rng.shuffle(names)
        rng.shuffle(motors)
        rename, remap = dict(zip(values, names)), dict(zip(actions, motors))
        renamed = tuple((rename[x], remap[act], rename[y]) for x, act, y in samples)
        old, new = induce(samples), induce(renamed)
        self.assertEqual(len(old.classes), len(new.classes))
        self.assertIsNotNone(discover_alignment(old, new))

    def test_unknown_varying_world_sizes(self):
        for n, m, aliases in ((3, 2, 2), (5, 3, 4), (7, 2, 3)):
            with self.subTest(n=n, m=m, aliases=aliases):
                a, b, _ = make_worlds(n + m, n, m, aliases)
                x, y = induce(observe(a, 4500, 34)), induce(observe(b, 4500, 35))
                self.assertEqual(len(x.classes), n)
                self.assertIsNotNone(discover_alignment(x, y))

    def test_invalid_input_contracts(self):
        with self.assertRaises(ValueError):
            make_worlds(0, states=9)
        with self.assertRaises(ValueError):
            observe(make_worlds(0)[0], 0, 2)
        with self.assertRaises(ValueError):
            induce([("a", "b", "a")], min_samples_per_condition=0)
        with self.assertRaises(ValueError):
            multiseed(runs=1)
        with self.assertRaises(ValueError):
            prediction_loss((("a", "b", "a"),), (("a", "b", "a"),), None, 0)
        with self.assertRaises(InsufficientEvidence):
            prediction_loss((("a", "b", "a"),), (("a", "b", "unknown"),), None)
        big = Representation(("x",), tuple((str(i),) for i in range(8)),
                             tuple((i,) for i in range(8)))
        with self.assertRaises(ValueError):
            discover_alignment(big, big)

    def test_seeded_reproducibility_and_export(self):
        self.assertEqual(single_trial(2, 2600, 150), single_trial(2, 2600, 150))
        model = representation_snapshot(2, 2600)
        self.assertEqual(len(model["source"]["invented_state_symbols"]), 7)
        self.assertEqual(len(model["discovered_graph_isomorphism"]["actions"]), 3)
        json.dumps(model)
        result = multiseed(0, 3, 2600, 150)
        self.assertEqual(result["exact_partitions_both_worlds"], 3)
        self.assertEqual(result["unrelated_graphs_rejected"], 3)

    def test_cli_emits_real_learned_model(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, "-m", "genesis", "emerge", "--runs", "1",
                       "--train-steps", "2600", "--test-steps", "100",
                       "--seed", "4", "--output", folder]
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads(Path(folder, "summary.json").read_text())
            model = json.loads(Path(folder, "invented_representation.json").read_text())
            self.assertEqual(summary["induced_states_left"], 7)
            self.assertEqual(len(model["source"]["classes_of_opaque_observations"]), 7)


if __name__ == "__main__":
    unittest.main()
