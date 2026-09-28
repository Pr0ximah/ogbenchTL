"""Cube manipulation tasks with optional temporal-logic region overlays."""

import colorsys

import mujoco
import numpy as np
from dm_control import mjcf

from ogbenchTL.manipspace.envs.cube_env import CubeEnv


class TLManipEnv(CubeEnv):
    """A :class:`CubeEnv` with non-colliding spherical region visualizations.

    Regions are deliberately visual-only. All cube task generation, observations,
    rewards, success checks, and dynamics are inherited unchanged from
    :class:`CubeEnv`.
    """

    _OBS_XYZ_CENTER = np.array([0.425, 0.0, 0.0])
    _OBS_XYZ_SCALER = 10.0

    def __init__(
        self,
        region_sample_seed=0,
        region_num=0,
        region_type="sphere",
        nonoverlapping_regions=True,
        sample_regions_func=None,
        sample_regions_params=None,
        override_regions=None,
        env_type="single",
        permute_blocks=True,
        *args,
        **kwargs,
    ):
        """Initialize a cube task and its optional region overlays.

        ``set_regions`` accepts regions in state-observation coordinates. Their
        centers and radii are mapped to world coordinates using the same cube
        observation convention: ``world_xyz = obs_xyz / 10 + [0.425, 0, 0]``.
        Sampled and overridden regions retain the historical world-coordinate
        convention.
        """
        if not isinstance(region_num, (int, np.integer)) or region_num < 0:
            raise ValueError("region_num must be a nonnegative integer.")
        if region_type != "sphere":
            raise ValueError("Only spherical TL regions are supported.")

        self._region_sample_seed = region_sample_seed
        self._region_num = region_num
        self._region_type = region_type
        self._nonoverlapping_regions = nonoverlapping_regions
        self._regions = None if override_regions is None else self._validate_regions(override_regions)
        self._regions_obs = None
        self._region_geoms_list = []
        self._sample_regions_func = sample_regions_func
        self._sample_regions_params = {
            "safe_margin": 0.1,
            "r_range": (0.05, 0.2),
            "alpha": 0.6,
        }
        if sample_regions_params is not None:
            self._sample_regions_params.update(sample_regions_params)

        super().__init__(
            env_type=env_type,
            permute_blocks=permute_blocks,
            *args,
            **kwargs,
        )

    @staticmethod
    def _validate_regions(regions):
        """Copy and validate a complete update before touching simulation state."""
        result = []
        for x, y, z, r, rgba in regions:
            values = np.asarray((x, y, z, r), dtype=float)
            color = np.asarray(rgba, dtype=float)
            if (not np.all(np.isfinite(values)) or values[3] <= 0
                    or color.shape != (4,) or not np.all(np.isfinite(color))
                    or np.any((color < 0) | (color > 1))):
                raise ValueError("Regions require finite centers, positive radii and RGBA in [0, 1].")
            result.append((*values.tolist(), tuple(color.tolist())))
        return result

    def _generate_colors(self, n, s_range=(0.7, 1.0), v_range=(0.7, 1.0)):
        golden_ratio_conjugate = (1 + np.sqrt(5)) / 2 - 1
        return [
            colorsys.hsv_to_rgb(
                (i * golden_ratio_conjugate) % 1,
                np.random.uniform(*s_range),
                np.random.uniform(*v_range),
            )
            for i in range(n)
        ]

    def set_regions(self, regions):
        """Set visual regions supplied in normalized observation coordinates."""
        self._regions_obs = self._validate_regions(regions)
        self._regions = [
            (
                *(
                    np.asarray((x, y, z)) / self._OBS_XYZ_SCALER
                    + self._OBS_XYZ_CENTER
                ).tolist(),
                r / self._OBS_XYZ_SCALER,
                rgba,
            )
            for x, y, z, r, rgba in self._regions_obs
        ]
        self._region_num = len(self._regions)

        current_capacity = len(self._region_geoms_list)
        if self._region_num != current_capacity:
            # Geometry count is fixed at compilation time; rebuild it on reset.
            self._mjcf_model = None
            self.mark_dirty()
            return

        if self._model is not None:
            for i, (x, y, z, r, rgba) in enumerate(self._regions):
                body_id = self._model.body(f"region_{i}").id
                geom_id = self._model.geom(f"region_{i}").id
                self._model.body_pos[body_id] = (x, y, z)
                mocap_id = self._model.body_mocapid[body_id]
                self._data.mocap_pos[mocap_id] = (x, y, z)
                # Keep the source model in sync for subsequent recompilation.
                if self._mjcf_model is not None:
                    self._mjcf_model.find("body", f"region_{i}").pos = (x, y, z)
                    geom = self._mjcf_model.find("geom", f"region_{i}")
                    geom.size = (r, 0, 0)
                    geom.rgba = rgba
                self._model.geom_size[geom_id] = (r, 0, 0)
                self._model.geom_rgba[geom_id] = rgba
            mujoco.mj_forward(self._model, self._data)

    def _sample_regions(self, region_num):
        if self._regions is not None:
            return self._regions
        if self._region_type != "sphere":
            raise NotImplementedError(f"Region type {self._region_type} not implemented.")
        if self._sample_regions_func is not None:
            self._regions = self._validate_regions(self._sample_regions_func(region_num))
            return self._regions

        if region_num == 0:
            self._regions = []
            return self._regions
        safe_margin = float(self._sample_regions_params["safe_margin"])
        alpha = float(self._sample_regions_params["alpha"])
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1].")
        workspace_bound = self._workspace_bounds
        lower = workspace_bound[0] + safe_margin
        upper = workspace_bound[1] - safe_margin
        workspace_size_avg = np.mean(workspace_bound[1] - workspace_bound[0])
        r_range = np.asarray(self._sample_regions_params["r_range"]) * workspace_size_avg

        if not np.isfinite(safe_margin) or safe_margin < 0 or np.any(lower > upper):
            raise ValueError("safe_margin must leave a nonempty sampling workspace.")
        if (r_range.shape != (2,) or not np.all(np.isfinite(r_range))
                or r_range[0] <= 0 or r_range[0] > r_range[1] or r_range[1] <= 0.02):
            raise ValueError("r_range must be positive, ordered, and allow world radii > 0.02.")

        rng_state = np.random.get_state()
        np.random.seed(self._region_sample_seed)
        try:
            colors = self._generate_colors(region_num)
            regions = []
            max_attempts = 10000
            for _ in range(max_attempts):
                pos = np.random.uniform(lower, upper)
                r = np.random.uniform(*r_range)
                r = min(r, *(pos - workspace_bound[0]), *(workspace_bound[1] - pos))
                if r <= 0.02:
                    continue
                if self._nonoverlapping_regions and any(
                    np.linalg.norm(pos - np.asarray(other[:3])) < r + other[3]
                    for other in regions
                ):
                    continue
                rgba = (*colors[len(regions)], self._sample_regions_params["alpha"])
                regions.append((*pos, r, rgba))
                if len(regions) == region_num:
                    break
            else:
                raise ValueError(
                    f"Could not sample {region_num} regions in {max_attempts} attempts; "
                    "reduce region_num/r_range or disable nonoverlapping_regions."
                )
            self._regions = regions
        finally:
            np.random.set_state(rng_state)
        return self._regions

    def get_regions_info(self):
        """Return regions in normalized state-observation coordinates."""
        if self._regions_obs is not None:
            return list(self._regions_obs)
        if self._regions is None:
            return []
        return [
            (
                *((np.asarray((x, y, z)) - self._OBS_XYZ_CENTER) * self._OBS_XYZ_SCALER),
                r * self._OBS_XYZ_SCALER,
                rgba,
            )
            for x, y, z, r, rgba in self._regions
        ]

    def add_objects(self, arena_mjcf):
        """Build the regular cube scene, then add visual-only region spheres."""
        super().add_objects(arena_mjcf)
        regions = self._sample_regions(self._region_num)

        self._region_geoms_list = []
        for i, (x, y, z, r, rgba) in enumerate(regions):
            region_mjcf = mjcf.from_path(
                (self._desc_dir / "region_sphere.xml").as_posix()
            )
            region_mjcf.find("body", "region_template").pos = (x, y, z)
            region_mjcf.find("body", "region_template").name = f"region_{i}"
            geom = region_mjcf.find("geom", "region_geom")
            geom.name = f"region_{i}"
            geom.size = (r, 0, 0)
            geom.rgba = rgba
            arena_mjcf.include_copy(region_mjcf)
            self._region_geoms_list.append(
                arena_mjcf.find("body", f"region_{i}").find_all("geom")
            )
