# rlx: Simple Implementations of Deep Reinforcement Learning with JAX Flax NNX
<a><img src="docs/rlx.png" width="240" align="right"/></a>

This repository provides high-performing implementations of PPO using JAX, Flax NNX and MuJoCo Playground. Unlike other framework it uses single-file implementations and the more readble Flax NNX API instead of Flax Linen. 

**Features:**
- Single-file implementation of DRL baselines with Flax NNX.
- Physics simulation, neural network inference and gradient computation all runs on GPU with `jit`-acceleration.
- Checkpointing with [orbax-checkpoint](https://orbax.readthedocs.io/)
- Config management with [hydra](https://hydra.cc)
- Logging with [wandb](https://wandb.ai/)
- Dependency management with [uv](https://docs.astral.sh/uv/) 

**Disclaimer 1:** This repository is not actively developed framework and will not provide any further support or documentation. It is intended for hobbyists and as a look-up for the usage of Flax NNX in DRL. For reliable and widely-validated and tested results, we recommend more mature frameworks e.g. [stable-baselines3](https://github.com/DLR-RM/stable-baselines3), [BRAX](https://github.com/google/brax), [CleanRL](https://github.com/vwxyzjn/cleanrl) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl). Nevertheless, we provide some benchmarking and fun training examples to validate the functionality of the implemented algorithms.

**Disclaimer 2:** Unlike `Brax` this implementation here vectorize with `vmap` instead of `pmap` and do not support multi-GPU usage. Which might be not required anyway for most users.

Nevertheless, feel free to submit issues and PR. While we cannot promise to integrate them, it might be helpful for other users.


## Getting started
Setup training environment with `uv`.
```bash
git clone git@github.com:alexanderdittrich/rlx.git
cd rlx 
uv sync
```

Run training:
```bash
uv run scripts/playground_ppo_train.py env_id=Go1JoystickWalk num_train_steps=200000000
```

## Benchmarks
...


## Passive Viewer - Visualization
...