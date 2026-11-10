"""Regressions and falsifications for stochastic, partially observable learning."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from genesis.partial import (
    Episode, PredictiveModel, make_world, multiseed, records,
    scramble_histories, train_select, trial,
)


class PartialObservabilityTests(unittest.TestCase):
    def test_opaque_string_and_vector_modalities(self):
        for mode in ('hex', 'vector'):
            w = make_world(17, mode)
            self.assertEqual(len(w.observation_tokens), 3)
            self.assertEqual(len(w.action_tokens), 2)
            self.assertTrue(all(isinstance(t, str if mode == 'hex' else tuple)
                                for t in w.observation_tokens))
            self.assertFalse(set(w.action_tokens) & set(w.observation_tokens))
            with self.assertRaises(ValueError): make_world(17, 'image')

    def test_partial_observation_and_stochasticity(self):
        w = make_world(3)
        self.assertTrue(all(0 < p < 1 for row in w.emissions for p in row))
        self.assertTrue(all(0 < p < 1 for state in w.transitions
                            for action in state for p in action))
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-12 for row in w.emissions))
        self.assertTrue(all(abs(sum(action) - 1.0) < 1e-12
                            for state in w.transitions for action in state))

    def test_generated_trajectories_are_seeded_and_have_no_hidden_labels(self):
        world = make_world(2, 'vector')
        a = world.rollout(5, 3, 40)
        self.assertEqual(a, world.rollout(5, 3, 40))
        self.assertNotEqual(a, world.rollout(6, 3, 40))
        self.assertTrue(all(isinstance(ep, Episode) for ep in a))
        self.assertEqual(len(records(a, 3)), 120)
        with self.assertRaises(ValueError): Episode(('one',), ('act',))
        with self.assertRaises(ValueError): world.rollout(0, 0, 10)
        with self.assertRaises(ValueError): records(a, 7)

    def test_contexts_do_not_cross_episode_boundaries(self):
        episodes = (Episode(('a', 'b', 'c'), ('x', 'y')),
                    Episode(('d', 'e', 'f'), ('x', 'y')))
        data = records(episodes, 2)
        self.assertEqual(data[0], (('x', 'a', None, None, None, None), 'b'))
        self.assertEqual(data[1], (('y', 'b', 'x', 'a', None, None), 'c'))
        self.assertEqual(data[2][0][2:], (None, None, None, None))

    def test_probability_distribution_is_normalized(self):
        train = records(make_world(6).rollout(7, 8, 100))
        model = PredictiveModel(2).fit(train)
        for context, _ in train[0:50]:
            dist = model.probabilities(context)
            self.assertAlmostEqual(sum(dist.values()), 1.0)
            self.assertTrue(all(0 < p < 1 for p in dist.values()))
        with self.assertRaises(ValueError): PredictiveModel(-1)
        with self.assertRaises(ValueError): model.loss([])
        with self.assertRaises(ValueError): PredictiveModel(2).fit([])

    def test_history_advantage_against_memorization_and_shuffling(self):
        for modality in ('hex', 'vector'):
            result = trial(0, modality)
            self.assertGreater(result['selected_history_depth'], 0)
            self.assertGreater(result['gain_vs_raw'], .02)
            self.assertGreater(result['gain_vs_shuffled'], .02)
            self.assertGreater(result['representation']['predictive_state_count'], 2)

    def test_iid_hidden_state_reset_negative_control(self):
        for modality in ('hex', 'vector'):
            result = trial(0, modality, True)
            self.assertEqual(result['selected_history_depth'], 0)
            self.assertAlmostEqual(result['gain_vs_raw'], 0.0, places=6)

    def test_symbol_renaming_does_not_change_scores(self):
        world = make_world(13)
        train = records(world.rollout(555, 10, 120))
        mapping = {v: f'new-token-{i}' for i, v in enumerate(sorted(
            {v for ctx, y in train for v in ctx if v is not None} |
            {y for _, y in train}, key=repr))}
        converted = [(tuple(mapping[v] if v is not None else None for v in ctx),
                      mapping[y]) for ctx, y in train]
        first = PredictiveModel(2).fit(train)
        second = PredictiveModel(2).fit(converted)
        self.assertAlmostEqual(first.loss(train), second.loss(converted), places=12)
        self.assertEqual(first.representation()['predictive_state_count'],
                         second.representation()['predictive_state_count'])

    def test_validation_not_test_selects_depth(self):
        world = make_world(2)
        training = records(world.rollout(999, 12, 110))
        validation = records(world.rollout(998, 5, 110))
        model, scores = train_select(training, validation)
        test = records(world.rollout(997, 5, 110))
        self.assertEqual(model.depth, train_select(training, validation)[0].depth)
        self.assertEqual(len(scores), 4)
        self.assertAlmostEqual(model.loss(test), model.loss(tuple(test)), places=12)
        scrambled = scramble_histories(training, 1)
        self.assertTrue(all(a[0][:2] == b[0][:2] and a[1] == b[1]
                            for a, b in zip(training, scrambled)))
        self.assertEqual(scrambled, scramble_histories(training, 1))

    def test_reproducible_multiseed_and_controls(self):
        result = multiseed(runs=2, training_episodes=6, steps=85)
        self.assertEqual(result, multiseed(runs=2, training_episodes=6, steps=85))
        self.assertEqual(result['total_experiments'], 8)
        self.assertEqual(set(result['groups']), {'hex:memory', 'hex:iid-control',
                                                 'vector:memory', 'vector:iid-control'})
        json.dumps(result)
        with self.assertRaises(ValueError): multiseed(runs=1)

    def test_cli_creates_reproducible_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            process = subprocess.run([sys.executable, '-m', 'genesis', 'partial',
                                      '--runs', '2', '--episodes', '6', '--steps', '85',
                                      '--seed', '3', '--output', folder],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            saved = json.loads(Path(folder, 'summary.json').read_text())
            self.assertEqual(saved['total_experiments'], 8)
            self.assertEqual(len(Path(folder, 'trials.jsonl').read_text().splitlines()), 8)


if __name__ == '__main__':
    unittest.main()
