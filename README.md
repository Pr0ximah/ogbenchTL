<div align="center">
    <img src="assets/ogbenchtl.svg" width="300px"/>
</div>

<div align="center" style="margin-bottom: 20px">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=grey&labelColor=%23fef7e7&color=%23FACD60"/>
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge&logo=github&logoColor=grey&labelColor=%23fef7e7&color=%23419156"/>
</div>

# OGBench-TL: Benchmarking Offline Goal-Conditioned RL with Temporal Logic Tasks Visualization Enhancements

<p align="center">
    <img src="assets/env_teaser.png" alt="OGBench-TL Environments" width="800"/>
</p>

## Overview

OGBench-TL is a **visualization-enhanced** version of [OGBench](https://github.com/seohongpark/ogbench) that adds specialized **Temporal Logic (TL) task visualization capabilities**.

**Compatibility**: Standard environments retain their inherited task logic. This fork also adds the custom `antmaze-easy-v0` layout and `tl-manip-v0`, a CubeEnv subclass with spherical region overlays. TL regions are visual annotations; this package does not evaluate temporal-logic formulas or define TL rewards. See the migration notes below for changes to the earlier TLManipEnv interface.

The key visualization enhancements include:
- **Temporal Logic Region Marking**: Visual representation of regions defined by temporal logic specifications
- **Waypoint Indicators**: Dynamic visualization of subgoals and waypoints for hierarchical planning
- **Customizable Background**: Optional white background for cleaner visualization
- **Simplified Maze Goal Visualization**: Original maze goal markers are hidden; cube target visibility follows CubeEnv settings

<p align="center" style="margin-top: 1.5rem;">
    <img src="assets/region_wp_example.png" alt="Temporal Logic Visualization Example" width="400"/>
</p>
<p align="center">
    <em>Temporal Logic task visualization example with <strong>blue circular waypoints</strong> and <strong>rectangular labeled regions</strong></em>
</p>

## Features

### Core Features (inherited from OGBench)
- 8 types of realistic and diverse environments:
  - **Locomotion**: PointMaze, AntMaze, HumanoidMaze, and AntSoccer
  - **Manipulation**: Cube, Scene, and Puzzle  
  - **Drawing**: Powderworld
- 85 datasets covering various challenges in offline goal-conditioned RL
- 410 tasks for standard (i.e., non-goal-conditioned) offline RL
- Support for both pixel-based and state-based observations
- Clean, well-tuned reference implementations of 6 offline goal-conditioned RL algorithms
- Fully reproducible scripts for benchmarking and dataset generation
- `pip`-installable, easy-to-use APIs based on Gymnasium
- No major dependencies other than MuJoCo

### Temporal Logic Visualization Enhancements
- **Region Visualization**: Circular and rectangular regions with customizable colors and positions
- **Waypoint Indicators**: Real-time subgoal visualization with customizable color coding
- **White Background**: Optional clean background for improved visual clarity
- **Dynamic Region Management**: Runtime addition and removal of regions
- **Simplified Maze Goal Display**: Original maze goal markers are hidden (TL tasks explicitly represent endpoints)

**Note**: All visualization features are purely additive and do not modify underlying tasks, rewards, or dataset structure.

## Installation

OGBench-TL can be installed directly from source:

```bash
git clone https://github.com/Pr0ximah/ogbenchTL.git
cd ogbenchTL
pip install -e .
```

It requires Python 3.10+ and has only three dependencies: `mujoco >= 3.1.6`, `dm_control >= 1.0.20`, and `gymnasium`.

## Usage for Temporal Logic Tasks

### Creating Environments

> [!CAUTION]
> Use `ogbenchTL.make_env_and_datasets` with a **dataset name** (e.g. `antmaze-large-navigate-v0`). For environment-only creation, set `env_only=True`. Alternatively, after importing `ogbenchTL`, use `gymnasium.make` with a **registered environment ID** (e.g. `antmaze-large-v0` or `tl-manip-v0`). The factory also accepts `tl-manip-v0` as an alias for `tl-manip-play-v0`.

Create environments with TL visualization support:

```python
import ogbenchTL

env, train_dataset, val_dataset = ogbenchTL.make_env_and_datasets(
    'antmaze-large-navigate-v0',
    region_type='circle',          # or 'rect' for rectangular regions  
    max_regions=5,                 # maximum number of regions to visualize
    white_bg=True,                 # enable white background
    wp_indicator=True              # enable waypoint indicators
)
```

> [!NOTE]
> The following options and region methods apply to **maze environments**. Temporal Logic visualization is **purely visual** and does not affect the underlying task logic or reward structure.
> 
> * The original goal/target markers have been **removed** since TL tasks can explicitly represent endpoints through regions and waypoints
> * Setting `max_regions=0`, `wp_indicator=False`, and `white_bg=False` makes the environment behave exactly like the original OGBench but no goal/target markers will be shown.
> * Region type must be consistent throughout the environment lifetime - dynamic switching between circle and rectangle regions is not supported
> * The total number of regions cannot exceed the `max_regions` value specified during environment creation

> [!CAUTION]
> When using `ogbenchTL.make_env_and_datasets`, access region methods through `env.unwrapped` (e.g., `env.unwrapped.add_region()`).
> Direct access like `env.add_region()` may not work due to environment wrappers.

### Adding Temporal Logic Regions

Once you have an environment instance, you can dynamically add regions:

```python
# Add a circular region
region1 = {
    'shape': 'circle',
    'center': [5.0, 3.0],          # (x, y) coordinates
    'radius': 1.5,
    'color': [1.0, 0.0, 0.0, 0.7]  # RGBA color (red with transparency)
}
env.unwrapped.add_region(region1)

# Alternative: use a SEPARATE environment created with region_type='rect'.
rect_env = ogbenchTL.make_env_and_datasets(
    'antmaze-large-navigate-v0', env_only=True, region_type='rect', max_regions=1
)
region2 = {
    'shape': 'rect',
    'center': [8.0, 6.0],
    'size': [2.0, 3.0],            # width, height
    'color': [0.0, 1.0, 0.0, 0.7]  # green with transparency
}
rect_env.unwrapped.add_region(region2)
```

### Setting Waypoint Indicators

Visualize waypoints and subgoals:

```python
# Default color (green-blue for subgoals)
env.unwrapped.set_wp_indicator([4.0, 7.0])

# Custom color (e.g., red for final goals)
env.unwrapped.set_wp_indicator([10.0, 2.0], wp_color=[1.0, 0.4, 0.4, 1.0])
```

> [!NOTE]
> Default waypoint color: `[0.4, 1.0, 0.7, 1.0]` (green-blue)
> Use `env.unwrapped.reset_wp_indicator()` to hide the waypoint indicator

### Complete Example

```python
import ogbenchTL
import numpy as np

# Create environment with TL visualization
env, train_dataset, val_dataset = ogbenchTL.make_env_and_datasets(
    'antmaze-large-navigate-v0',
    region_type='circle',
    max_regions=3,
    white_bg=True,
    wp_indicator=True
)

# Define regions
regions = [
    {'shape': 'circle', 'center': [3.0, 2.0], 'radius': 1.2, 'color': [1.0, 0.0, 0.0, 0.7]},
    {'shape': 'circle', 'center': [8.0, 5.0], 'radius': 1.5, 'color': [0.0, 0.0, 1.0, 0.7]},
    # Note: All regions must be circles since region_type='circle'
]

# Add regions
for region in regions:
    env.unwrapped.add_region(region)

# env.unwrapped.clear_regions()  # Example of clearing all regions

# Set a waypoint indicator with default color
env.unwrapped.set_wp_indicator([7.0, 4.0])

# Reset environment with specific task
ob, info = env.reset(options={'task_id': 1})

# Render the environment to see regions and waypoints
frame = env.render()

# Step through environment
done = False
while not done:
    action = env.action_space.sample()
    ob, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    frame = env.render()  # Visualize current state with regions and waypoints
```

## Environment Structure

### Locomotion Environments with TL Support
- `PointMaze`: Simple point mass navigation with TL region marking
- `AntMaze`: Ant robot navigation with complex terrain and TL visualization  
- `HumanoidMaze`: Humanoid robot navigation with full-body dynamics and TL support
- `AntSoccer`: Ant robot soccer environment (TL features planned)

### Custom Easy AntMaze

`antmaze-easy-v0` provides an 8×8 custom maze, five start/goal tasks, and a 1000-step episode limit:

```python
env = ogbenchTL.make_env_and_datasets('antmaze-easy-navigate-v0', env_only=True)
ob, info = env.reset(seed=42, options={'task_id': 1})
```

This repository does not bundle an easy dataset. To load data, provide matching local `dataset_path` and `-val.npz` files; the environment name alone does not establish availability on the upstream dataset server.

### Cube Manipulation with TL Regions

`tl-manip-v0` extends CubeEnv with non-colliding spherical overlays. Its task generation, state observations, rewards, success checks, and dynamics come from CubeEnv. It supports `env_type='single'`, `'double'`, `'triple'`, `'quadruple'`, and `'octuple'`. The registered TL environment has a **2000-step** limit, which differs from standard Cube registrations.

```python
import ogbenchTL

env = ogbenchTL.make_env_and_datasets(
    'tl-manip-v0', env_only=True, env_type='single',
    terminate_at_goal=False,  # useful when an external TL controller decides completion
)
# Each region is (x, y, z, radius, rgba), in normalized state coordinates.
env.unwrapped.set_regions([(0.0, 0.0, 1.0, 0.5, (0.2, 0.5, 1.0, 0.4))])
ob, info = env.reset(seed=42, options={'task_id': 1})

# Same count: position, radius and color update immediately.
env.unwrapped.set_regions([(1.0, 0.0, 1.0, 0.4, (1.0, 0.3, 0.2, 0.4))])
regions = env.unwrapped.get_regions_info()

# Different count (including clearing): call reset to rebuild the scene.
env.unwrapped.set_regions([])
ob, info = env.reset(seed=42, options={'task_id': 1})
```

- `set_regions()` and `get_regions_info()` use normalized state coordinates: `world_xyz = obs_xyz / 10 + [0.425, 0, 0]`, `world_radius = obs_radius / 10`.
- `override_regions` and `sample_regions_func(region_num)` use **world coordinates**. Automatic sampling uses `region_num`, `region_sample_seed`, and `nonoverlapping_regions`.
- `sample_regions_params` contains `safe_margin` (world units), `r_range` (fractions of mean workspace extent), and `alpha`. Defaults are `0.1`, `(0.05, 0.2)`, and `0.6`. Invalid ranges or failure to place all regions within 10,000 attempts raise `ValueError`.
- For states, the factory maps `tl-manip-play-v0` (or `tl-manip-v0`) to `cube-{env_type}-play-v0`. `dataset_only=True` uses `cur_env` when supplied, otherwise `env_type` (default `single`). Pixel data requires an explicit matching `dataset_path`; environment-only pixel use is supported.
- `visualize_info` controls inherited cube target displays. Cameras, arm transparency, and ghost-arm rendering belong to the manipulation base environment, not exclusively to TLManipEnv.
- Scene and Puzzle TL integration remains future work.

**Migration from the earlier TLManipEnv:** the previous arm-only state had 19 values and incomplete reward/success methods. The default single-cube state now has 28 values (19 arm values plus 9 cube values); multiple cubes use `19 + 9 * num_cubes`. Old 19-dimensional checkpoints are not directly compatible. Cube success may terminate an episode unless `terminate_at_goal=False`; region entry does not itself produce reward or TL success.

### Regression Tests

From the repository root, with dependencies installed:

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -v
```

Tests cover easy tasks, cube dynamics/observation parity, region updates and rebuilds, bounded sampling, and dataset routing. Dataset routing tests mock file loading and downloads; they do not fetch benchmark data.

## Planned Features

Task definitions currently remain aligned with CubeEnv. We may revise them in a future version to better meet temporal-logic task requirements; the design is not yet decided. This release does not introduce TL-specific task definitions, rewards, or success criteria.

- **Additional Manipulation TL Integration**: Extend region visualization to Scene and Puzzle
- **Multi-agent Components**: Add support for multi-agent temporal logic coordination tasks and visualization
- **Complex Region Shapes**: Support for polygonal and custom-shaped regions

## More documentation

Please refer to the original [OGBench Repo](https://github.com/seohongpark/ogbench) for more detailed documentation on core functionalities, algorithms, and dataset generation.

## Acknowledgments

This fork builds on [OGBench](https://github.com/seohongpark/ogbench), adding TL visualization utilities and a custom easy maze. Cube region overlays preserve the inherited CubeEnv task logic.

This library inherits code from:
- [D4RL](https://github.com/Farama-Foundation/D4RL) for dataset structure and AntMaze environment
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and [dm_control](https://github.com/google-deepmind/dm_control) for locomotion agents  
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for robot descriptions
- [Meta-World](https://github.com/Farama-Foundation/Metaworld) for manipulation objects
- [Powderworld](https://github.com/kvfrans/powderworld) for the drawing environment

Special thanks to the original OGBench authors for providing an excellent foundation.

## Citation

If you use OGBench-TL in your research, please cite both the original OGBench paper and acknowledge this extension:

```bibtex
@inproceedings{ogbench_park2025,
  title={OGBench: Benchmarking Offline Goal-Conditioned RL},
  author={Park, Seohong and Frans, Kevin and Eysenbach, Benjamin and Levine, Sergey},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
