"""Run with MUJOCO_GL=egl python -m unittest discover -s tests -v."""
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

import ogbenchTL
from ogbenchTL.manipspace.envs.cube_env import CubeEnv
from ogbenchTL.manipspace.envs.tl_manip_env import TLManipEnv


class RegionTests(unittest.TestCase):
    def test_live_update_reset_recompile_and_resize(self):
        env = TLManipEnv(region_num=1)
        env.reset(seed=42, options={'task_id': 1})
        regions = [(1., 1., 1., .5, (1., 0., 0., .5))]
        env.set_regions(regions)
        def check():
            np.testing.assert_allclose(env.data.body('region_0').xpos, [.525, .1, .1])
            np.testing.assert_allclose(env.model.geom('region_0').size[0], .05)
            np.testing.assert_allclose(env.model.geom('region_0').rgba, regions[0][4])
        check()
        env.reset(seed=42, options={'task_id': 1})
        check()
        env.mark_dirty()
        env.reset(seed=42, options={'task_id': 1})
        check()
        env.set_regions(regions * 2)
        env.reset(seed=42, options={'task_id': 1})
        self.assertEqual(len(env._region_geoms_list), 2)
        env.set_regions([])
        env.reset(seed=42, options={'task_id': 1})
        self.assertEqual(len(env._region_geoms_list), 0)
        self.assertEqual(env.get_regions_info(), [])

    def test_cube_parity_with_regions(self):
        for variant in ('single', 'double'):
            with self.subTest(variant=variant):
                cube = CubeEnv(env_type=variant)
                tl = TLManipEnv(env_type=variant, region_num=3)
                # Supply identical random settling actions used by CubeEnv reset.
                with patch('gymnasium.spaces.Box.sample', return_value=np.zeros(5)):
                    a, ai = cube.reset(seed=42, options={'task_id': 1})
                    b, bi = tl.reset(seed=42, options={'task_id': 1})
                np.testing.assert_allclose(a, b)
                np.testing.assert_allclose(ai['goal'], bi['goal'])
                for action in np.random.RandomState(12).uniform(-.3, .3, (5, 5)):
                    a = cube.step(action)
                    b = tl.step(action)
                    np.testing.assert_allclose(a[0], b[0], atol=1e-10)
                    self.assertEqual(a[1:4], b[1:4])
                for i in range(3):
                    geom = tl.model.geom(f'region_{i}')
                    self.assertEqual(geom.contype[0], 0)
                    self.assertEqual(geom.conaffinity[0], 0)

    def test_sampling_reproducibility_and_rng_isolation(self):
        env = TLManipEnv(region_sample_seed=11)
        state = np.random.get_state()
        a = env._sample_regions(3)
        after = np.random.get_state()
        np.testing.assert_equal(state, after)
        b = TLManipEnv(region_sample_seed=11)._sample_regions(3)
        np.testing.assert_allclose(np.array([r[:4] for r in a]), np.array([r[:4] for r in b]))
        np.testing.assert_allclose(np.array([r[4] for r in a]), np.array([r[4] for r in b]))

    def test_impossible_sampling_fails_and_restores_rng(self):
        for params in (
            {'safe_margin': .1, 'r_range': (.01, .02), 'alpha': .6},
            {'safe_margin': .3, 'r_range': (.05, .2), 'alpha': .6},
            {'safe_margin': .1, 'r_range': (.2, .3), 'alpha': .6},
        ):
            env = TLManipEnv(sample_regions_params=params)
            state = np.random.get_state()
            with self.assertRaises(ValueError):
                env._sample_regions(100)
            np.testing.assert_equal(state, np.random.get_state())

    def test_pixel_render(self):
        env = TLManipEnv(ob_type='pixels', width=64, height=64, region_num=1)
        try:
            ob, _ = env.reset(seed=42, options={'task_id': 1})
            self.assertEqual(ob.shape, (64, 64, 3))
            self.assertEqual(ob.dtype, np.uint8)
            env.set_regions([(1., 1., 1., .5, (1., 0., 0., .5))])
            self.assertEqual(env.compute_observation().shape, ob.shape)
        finally:
            if env._renderer is not None:
                env._renderer.close()

    def test_custom_sampler_and_world_override(self):
        world = [(.425, .1, .1, .05, (1., 0., 0., .5))]
        for kwargs in ({'override_regions': world}, {'sample_regions_func': lambda n: world}):
            env = TLManipEnv(**kwargs)
            env.reset(seed=42, options={'task_id': 1})
            np.testing.assert_allclose(env.get_regions_info()[0][:4], [0., 1., 1., .5])
            self.assertEqual(len(env._region_geoms_list), 1)


class FactoryTests(unittest.TestCase):
    def test_environment_aliases(self):
        for name in ('tl-manip-v0', 'tl-manip-play-v0'):
            env = ogbenchTL.make_env_and_datasets(name, env_only=True)
            ob, _ = env.reset(seed=1, options={'task_id': 1})
            self.assertEqual(ob.shape, (28,))
            self.assertEqual(env.spec.id, 'tl-manip-v0')

    def test_dataset_routing_without_downloads(self):
        for variant in ('single', 'double', 'triple', 'quadruple', 'octuple'):
            with patch('ogbenchTL.utils.download_datasets') as download, patch(
                'ogbenchTL.utils.load_dataset', return_value={}
            ) as load:
                ogbenchTL.make_env_and_datasets('tl-manip-v0', dataset_only=True, env_type=variant)
                self.assertEqual(download.call_args.args[0], [f'cube-{variant}-play-v0'])
                self.assertIn(f'cube-{variant}-play-v0.npz', load.call_args_list[0].args[0])
        env = gym.make('tl-manip-v0', env_type='double')
        with patch('ogbenchTL.utils.download_datasets') as download, patch('ogbenchTL.utils.load_dataset', return_value={}):
            ogbenchTL.make_env_and_datasets('tl-manip-play-v0', dataset_only=True, cur_env=env)
            self.assertEqual(download.call_args.args[0], ['cube-double-play-v0'])

    def test_pixel_datasets_require_explicit_path(self):
        with self.assertRaisesRegex(ValueError, 'dataset_path'):
            ogbenchTL.make_env_and_datasets('tl-manip-v0', dataset_only=True, ob_type='pixels')
        with patch('ogbenchTL.utils.download_datasets') as download, patch('ogbenchTL.utils.load_dataset', return_value={}) as load:
            ogbenchTL.make_env_and_datasets('tl-manip-v0', dataset_only=True, ob_type='pixels', dataset_path='/tmp/custom.npz')
            download.assert_not_called()
            self.assertEqual(load.call_args.kwargs['ob_dtype'], np.uint8)

    def test_easy_tasks(self):
        env = ogbenchTL.make_env_and_datasets('antmaze-easy-navigate-v0', env_only=True)
        self.assertEqual(env.spec.max_episode_steps, 1000)
        for task_id in range(1, 6):
            ob, _ = env.reset(seed=42, options={'task_id': task_id})
            self.assertEqual(ob.shape, (29,))
            result = env.step(np.zeros(env.action_space.shape))
            self.assertTrue(np.all(np.isfinite(result[0])))


if __name__ == '__main__':
    unittest.main()
