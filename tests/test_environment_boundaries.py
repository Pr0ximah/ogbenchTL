"""Boundary and lifecycle regressions; no benchmark downloads required."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

import ogbenchTL
from ogbenchTL.manipspace.envs.tl_manip_env import TLManipEnv


class BoundaryTests(unittest.TestCase):
    def make(self, name, **kwargs):
        env = gym.make(name, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_waypoint_disabled(self):
        env = self.make('antmaze-easy-v0', wp_indicator=False, max_regions=0, white_bg=False)
        env.reset(seed=42)
        env.unwrapped.set_wp_indicator([0., 0.])
        env.unwrapped.reset_wp_indicator()
        env.step(np.zeros(8))
        self.assertFalse(env.unwrapped._wp_indicator)

    def test_pixel_sizes_and_close(self):
        for name in ('antmaze-easy-v0', 'tl-manip-v0'):
            for size in ({}, {'width': 80, 'height': 48}):
                with self.subTest(name=name, size=size):
                    env = self.make(name, ob_type='pixels', **size)
                    ob, info = env.reset(seed=42, options={'task_id': 1})
                    self.assertTrue(env.observation_space.contains(ob))
                    self.assertEqual(info['goal'].shape, ob.shape)
                    ob = env.step(np.zeros(env.action_space.shape))[0]
                    self.assertTrue(env.observation_space.contains(ob))
                    env.close()
                    attr = 'custom_renderer' if 'maze' in name else '_renderer'
                    self.assertIsNone(getattr(env.unwrapped, attr))
                    env.close()

    def test_tempfiles_and_descriptors_are_released(self):
        with tempfile.TemporaryDirectory() as directory:
            before = len(os.listdir('/proc/self/fd'))
            original = tempfile.mkstemp
            def create(**kwargs):
                return original(dir=directory, **kwargs)
            with patch('ogbenchTL.locomaze.maze.tempfile.mkstemp', side_effect=create):
                for _ in range(3):
                    env = self.make('antmaze-easy-v0')
                    env.close()
                # The temporary XML must also be removed when construction fails.
                with self.assertRaises(TypeError):
                    gym.make('antmaze-easy-v0', invalid_constructor_option=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(len(os.listdir('/proc/self/fd')), before)

    def test_rejected_regions_do_not_poison_scene(self):
        for shape in ('circle', 'rect'):
            env = self.make('antmaze-easy-v0', region_type=shape, max_regions=1)
            region = {'shape': shape, 'center': [1., 2.], 'radius': .5,
                      'size': [1., 2.], 'color': [1., 0., 0., .5]}
            for wrong in ({**region, 'shape': 'triangle'}, {**region, 'color': [1, 0]},
                          {**region, 'center': [np.nan, 0]}):
                with self.assertRaises(ValueError):
                    env.unwrapped.add_region(wrong)
                self.assertEqual(env.unwrapped._regions, [])
                env.unwrapped.clear_regions()
            env.unwrapped.add_region(region)
            # External mutation must not corrupt bookkeeping either.
            region['shape'] = 'triangle'
            env.unwrapped.clear_regions()
            self.assertEqual(env.unwrapped._regions, [])

    def test_seed_reproducibility(self):
        for name in ('antmaze-easy-v0', 'antmaze-medium-v0', 'antsoccer-arena-v0',
                     'tl-manip-v0', 'cube-single-v0'):
            for options in ({}, {'task_id': 1}):
                with self.subTest(name=name, options=options):
                    env = self.make(name)
                    ob1, info1 = env.reset(seed=42, options=options)
                    task1 = env.unwrapped.cur_task_id
                    env.step(np.zeros(env.action_space.shape))
                    # Global NumPy and action-space RNG must not influence reset.
                    np.random.uniform(size=17)
                    env.action_space.sample()
                    ob2, info2 = env.reset(seed=42, options=options)
                    np.testing.assert_allclose(ob1, ob2, atol=1e-10)
                    np.testing.assert_allclose(info1['goal'], info2['goal'], atol=1e-10)
                    self.assertEqual(task1, env.unwrapped.cur_task_id)

    def test_action_space_seed_persists(self):
        for name in ('tl-manip-v0', 'cube-single-v0'):
            env = self.make(name)
            env.action_space.seed(23)
            first = env.action_space.sample()
            env.action_space.seed(23)
            np.testing.assert_array_equal(first, env.action_space.sample())

    def test_invalid_tl_region_is_atomic_and_copied(self):
        env = self.make('tl-manip-v0', region_num=1)
        env.reset(seed=42)
        before = env.unwrapped.get_regions_info()
        before_pos = env.unwrapped.data.body('region_0').xpos.copy()
        for radius, color in ((-.1, [1, 0, 0, .5]), (.1, [1, 0]), (.1, [1, 0, np.nan, .5])):
            with self.assertRaises(ValueError):
                env.unwrapped.set_regions([(0, 0, 1, radius, color)])
            self.assertEqual(before, env.unwrapped.get_regions_info())
            np.testing.assert_array_equal(before_pos, env.unwrapped.data.body('region_0').xpos)
        external = [[0., 0., 1., .5, [1., 0., 0., .5]]]
        env.unwrapped.set_regions(external)
        external[0][0] = 99
        external[0][4][0] = 0
        snapshot = env.unwrapped.get_regions_info()
        self.assertEqual(snapshot[0][0], 0)
        self.assertEqual(snapshot[0][4][0], 1)
        snapshot.clear()
        self.assertEqual(len(env.unwrapped.get_regions_info()), 1)

    def test_partial_sampling_parameters(self):
        env = self.make('tl-manip-v0', region_num=2, sample_regions_params={'alpha': .3})
        env.reset(seed=42)
        for region in env.unwrapped.get_regions_info():
            self.assertEqual(region[4][3], .3)

    def test_oracle_from_real_npz(self):
        for variant, count in (('single', 1), ('double', 2)):
            with tempfile.TemporaryDirectory() as directory:
                qpos = np.zeros((4, 14 + 7 * count))
                expected = []
                for i in range(count):
                    xyz = np.array([.425 + i * .01, .1, .02])
                    qpos[:, 14 + 7*i:17 + 7*i] = xyz
                    expected.extend((xyz - [.425, 0, 0]) * 10)
                payload = dict(observations=np.zeros((4, 19 + 9 * count)),
                               actions=np.zeros((4, 5)), terminals=np.array([0, 0, 0, 1]), qpos=qpos)
                path = str(Path(directory) / 'cube.npz')
                np.savez(path, **payload)
                np.savez(path.replace('.npz', '-val.npz'), **payload)
                for only in (True, False):
                    with patch('ogbenchTL.utils.download_datasets') as download:
                        result = ogbenchTL.make_env_and_datasets(
                            'tl-manip-play-oraclerep-v0', dataset_path=path,
                            env_type=variant, dataset_only=only)
                        download.assert_not_called()
                    if not only:
                        env, train, val = result
                        self.addCleanup(env.close)
                        _, info = env.reset(seed=42)
                        self.assertEqual(info['goal'].shape, (3 * count,))
                    else:
                        train, val = result
                    for dataset in (train, val):
                        self.assertEqual(dataset['oracle_reps'].shape, (3, 3 * count))
                        np.testing.assert_allclose(dataset['oracle_reps'], np.tile(expected, (3, 1)), atol=1e-6)
                        self.assertNotIn('qpos', dataset)


if __name__ == '__main__':
    unittest.main()
