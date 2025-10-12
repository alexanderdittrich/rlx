# rlx
Baseline RL algorithms implemented in JAX Flax NNX.


## Why yet another repository for baseline RL algorithms?

While JAX can provide a significant speed up to computation, it is often not intuitive to use. This is in particular the case for Flax Linen API. This small framework provides a collection of reimplemented algorithms in the new API - Flax NNX. The single-file orientation of this repo is heavily inspired by the design philosophy of [CleanRL](https://github.com/vwxyzjn/cleanrl).

**Features:**
- (Almost) single-file implementation of DRL baselines.
- Checkpointing with [orbax-checkpoint](https://orbax.readthedocs.io/)
- Config management with [hydra](https://hydra.cc)
- Logging with [wandb](https://wandb.ai/)
- Dependency management with uv 

**Disclaimer:** This repository is not actively developed and will not provide any further support or documentation. For serious research, we definitely recommend mature frameworks e.g. ([BRAX](https://github.com/google/brax)) and ([RSL-RL](https://github.com/leggedrobotics/rsl_rl)).


## Benchmarks
- [ ] Discrete environments: `Acrobot-v1`, `CartPole-v1`, `MountainCar-v0`, `LunarLander-v3`
- [ ] Continuous environments: `Pendulum-v1`,`BipedalWalker-v3`,`HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`
- [ ] Vision: `CarRacing-v3`, `ALE/SpaceInvaders-v5`, `ALE/Breakout-v5`


## Roadmap:
- [ ] Playground MJX API. 
- [ ] `nnx.scan`-integration.
- [ ] Extensive benchmarking.
- [ ] External learn-API.
- [ ] Checkpointing and replay.
- [ ] Integration of further algorithms -> SAC, DreamerV3.