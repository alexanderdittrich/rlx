"""
SAC for MuJoCo Playground with JAX Flax NNX and Optax.

Portions of this file are adapted from Brax
(https://github.com/google/brax),
Copyright 2022 The Brax Authors,
licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0).

Changes:
Ported to Flax NNX; single-file-oriented code structure,
hydra-config management, obs normalization, vectorized envs,
numpy replay buffer with JAX-based updates.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx
from mujoco_playground import registry, wrapper
from rlx.common import running_statistics

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class SACConfig:
    env_id: str = "Go1JoystickFlatTerrain"
    num_envs: int = 128  # Brax SAC default

    total_timesteps: int = 20_000_000
    learning_starts: int = 8_192
    batch_size: int = 256

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005             # Polyak soft-update coefficient
    alpha: float = 0.2             # initial / fixed entropy coefficient
    auto_tune_alpha: bool = True   # auto-tune α via dual gradient descent
    target_entropy_scale: float = 1.0  # target_entropy = -action_dim * scale
    reward_scaling: float = 1.0

    buffer_size: int = 1_048_576   # Brax max_replay_size

    hidden_sizes: list[int] | None = None   # [256, 256] by default
    activation: str = "relu"

    policy_obs_key: str = "state"
    value_obs_key: str = "privileged_state"

    log_frequency: int = 1_000    # log every N global_steps
    eval_frequency: int = 10_000  # eval every N global_steps
    eval_episodes: int = 128
    checkpoint_frequency: int = 50_000  # 0 = off
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    use_checkpointing: bool = False
    wandb_project: str = "mjxrl_sac"
    verbose: bool = True
    progress_bar: bool = False
    use_domain_randomization: bool = False
    env_overrides: dict = field(default_factory=dict)

    seed: int = 42

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]
        self.num_iterations = self.total_timesteps // self.num_envs


# ── Networks ──────────────────────────────────────────────────────────────────

ACTIVATIONS = {
    "tanh": nnx.tanh,
    "relu": nnx.relu,
    "swish": nnx.swish,
    "elu": nnx.elu,
    "gelu": nnx.gelu,
}

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _mlp(in_size, hidden_sizes, out_size, activation, rngs) -> nnx.Sequential:
    ki = nnx.initializers.lecun_uniform()
    layers, cur = [], in_size
    for h in hidden_sizes:
        layers += [nnx.Linear(cur, h, kernel_init=ki, rngs=rngs), activation]
        cur = h
    layers.append(nnx.Linear(cur, out_size, kernel_init=ki, rngs=rngs))
    return nnx.Sequential(*layers)


class Actor(nnx.Module):
    """Squashed-Gaussian actor: obs → tanh(N(μ, σ)) with log-prob."""

    def __init__(self, obs_size, action_size, hidden_sizes, activation, rngs):
        self.net = _mlp(obs_size, hidden_sizes, 2 * action_size, activation, rngs)

    def __call__(self, obs: jax.Array):
        out = self.net(obs)
        loc, log_std = jnp.split(out, 2, axis=-1)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return loc, jnp.exp(log_std)

    def sample(self, obs: jax.Array, key: jax.Array):
        """Stochastic action + tanh-corrected log-prob. Used during training."""
        loc, std = self(obs)
        u = loc + std * jax.random.normal(key, loc.shape)
        action = jnp.tanh(u)
        # log π(a|s) = Σ[log N(u;μ,σ)] - Σ[log(1 - tanh²(u))]
        # Numerically stable tanh correction: 2*(log2 - u - softplus(-2u))
        log_prob_gaussian = (
            -0.5 * jnp.square((u - loc) / std)
            - jnp.log(std)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        ).sum(-1)
        tanh_correction = (
            2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        ).sum(-1)
        return action, log_prob_gaussian - tanh_correction

    def mode(self, obs: jax.Array) -> jax.Array:
        """Deterministic action for evaluation."""
        loc, _ = self(obs)
        return jnp.tanh(loc)


class QNetwork(nnx.Module):
    """Q(s, a) → scalar."""

    def __init__(self, obs_size, action_size, hidden_sizes, activation, rngs):
        self.net = _mlp(obs_size + action_size, hidden_sizes, 1, activation, rngs)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        return self.net(jnp.concatenate([obs, action], axis=-1)).squeeze(-1)


class Critics(nnx.Module):
    """Twin Q-networks for gradient computation via a single optimizer."""

    def __init__(self, qf1: QNetwork, qf2: QNetwork):
        self.qf1 = qf1
        self.qf2 = qf2


class Alpha(nnx.Module):
    """Learnable log-temperature for entropy auto-tuning."""

    def __init__(self):
        self.log_alpha = nnx.Param(jnp.array(0.0, dtype=jnp.float32))

    @property
    def value(self) -> jax.Array:
        return jnp.exp(self.log_alpha.value)


# ── Obs helpers ───────────────────────────────────────────────────────────────


def get_obs(obs: Any, key: str) -> jax.Array:
    return obs[key] if isinstance(obs, (dict, Mapping)) else obs


def normalize_obs(raw: jax.Array, stats) -> jax.Array:
    return running_statistics.normalize(raw, stats)


# ── Replay buffer ─────────────────────────────────────────────────────────────


class ReplayBuffer:
    """Circular numpy replay buffer with vectorised batch-add."""

    def __init__(
        self,
        buffer_size: int,
        actor_obs_size: int,
        critic_obs_size: int,
        action_size: int,
    ):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

        self.actor_obs = np.zeros((buffer_size, actor_obs_size), np.float32)
        self.critic_obs = np.zeros((buffer_size, critic_obs_size), np.float32)
        self.next_actor_obs = np.zeros((buffer_size, actor_obs_size), np.float32)
        self.next_critic_obs = np.zeros((buffer_size, critic_obs_size), np.float32)
        self.actions = np.zeros((buffer_size, action_size), np.float32)
        self.rewards = np.zeros(buffer_size, np.float32)
        self.dones = np.zeros(buffer_size, np.float32)

    def add_batch(
        self,
        actor_obs,
        critic_obs,
        next_actor_obs,
        next_critic_obs,
        actions,
        rewards,
        dones,
    ):
        n = len(actor_obs)
        idxs = np.arange(self.ptr, self.ptr + n) % self.buffer_size
        self.actor_obs[idxs] = actor_obs
        self.critic_obs[idxs] = critic_obs
        self.next_actor_obs[idxs] = next_actor_obs
        self.next_critic_obs[idxs] = next_critic_obs
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones
        self.ptr = (self.ptr + n) % self.buffer_size
        self.size = min(self.size + n, self.buffer_size)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idxs = rng.integers(0, self.size, size=batch_size)
        return {
            "actor_obs": jnp.array(self.actor_obs[idxs]),
            "critic_obs": jnp.array(self.critic_obs[idxs]),
            "next_actor_obs": jnp.array(self.next_actor_obs[idxs]),
            "next_critic_obs": jnp.array(self.next_critic_obs[idxs]),
            "actions": jnp.array(self.actions[idxs]),
            "rewards": jnp.array(self.rewards[idxs]),
            "dones": jnp.array(self.dones[idxs]),
        }


# ── Update functions ──────────────────────────────────────────────────────────
# All update functions operate on raw observations from the buffer and
# normalise internally with the current running stats — so old experiences are
# always re-normalised to the latest scale before computing losses.


@nnx.jit
def update_critic(
    actor: Actor,
    critics: Critics,
    qf1_target: QNetwork,
    qf2_target: QNetwork,
    critic_optimizer: nnx.Optimizer,
    raw_critic_obs: jax.Array,
    raw_next_actor_obs: jax.Array,
    raw_next_critic_obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    actor_stats,
    critic_stats,
    alpha: jax.Array,
    gamma: float,
    key: jax.Array,
) -> dict:
    critic_obs = normalize_obs(raw_critic_obs, critic_stats)
    next_actor_obs = normalize_obs(raw_next_actor_obs, actor_stats)
    next_critic_obs = normalize_obs(raw_next_critic_obs, critic_stats)

    def loss_fn(critics: Critics):
        next_actions, next_log_probs = actor.sample(next_actor_obs, key)
        q1_next = qf1_target(next_critic_obs, next_actions)
        q2_next = qf2_target(next_critic_obs, next_actions)
        min_q_next = jnp.minimum(q1_next, q2_next) - alpha * next_log_probs
        target_q = jax.lax.stop_gradient(
            rewards + (1.0 - dones) * gamma * min_q_next
        )

        q1 = critics.qf1(critic_obs, actions)
        q2 = critics.qf2(critic_obs, actions)
        loss = 0.5 * (
            jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)
        )
        return loss, {
            "loss/critic": loss,
            "diagnostics/q1_mean": q1.mean(),
            "diagnostics/q2_mean": q2.mean(),
        }

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(critics)
    critic_optimizer.update(model=critics, grads=grads)
    return metrics


@nnx.jit
def update_actor(
    actor: Actor,
    critics: Critics,
    actor_optimizer: nnx.Optimizer,
    raw_actor_obs: jax.Array,
    raw_critic_obs: jax.Array,
    actor_stats,
    critic_stats,
    alpha: jax.Array,
    key: jax.Array,
) -> dict:
    actor_obs = normalize_obs(raw_actor_obs, actor_stats)
    critic_obs = normalize_obs(raw_critic_obs, critic_stats)

    def loss_fn(actor: Actor):
        sampled_actions, log_probs = actor.sample(actor_obs, key)
        q1 = critics.qf1(critic_obs, sampled_actions)
        q2 = critics.qf2(critic_obs, sampled_actions)
        loss = jnp.mean(alpha * log_probs - jnp.minimum(q1, q2))
        return loss, {
            "loss/actor": loss,
            "diagnostics/entropy": -log_probs.mean(),
        }

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(actor)
    actor_optimizer.update(model=actor, grads=grads)
    return metrics


@nnx.jit
def update_alpha(
    actor: Actor,
    alpha_module: Alpha,
    alpha_optimizer: nnx.Optimizer,
    raw_actor_obs: jax.Array,
    actor_stats,
    target_entropy: float,
    key: jax.Array,
) -> dict:
    actor_obs = normalize_obs(raw_actor_obs, actor_stats)

    def loss_fn(alpha_module: Alpha):
        _, log_probs = actor.sample(actor_obs, key)
        alpha = jnp.exp(alpha_module.log_alpha.value)
        loss = jnp.mean(alpha * jax.lax.stop_gradient(-log_probs - target_entropy))
        return loss, {"loss/alpha": loss, "alpha": alpha}

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(alpha_module)
    alpha_optimizer.update(model=alpha_module, grads=grads)
    return metrics


@nnx.jit
def soft_update(
    critics: Critics,
    qf1_target: QNetwork,
    qf2_target: QNetwork,
    tau: float,
):
    """Polyak update: θ_target ← τ·θ + (1-τ)·θ_target."""
    new_qf1_target = jax.tree.map(
        lambda src, tgt: tau * src + (1.0 - tau) * tgt,
        nnx.state(critics.qf1),
        nnx.state(qf1_target),
    )
    new_qf2_target = jax.tree.map(
        lambda src, tgt: tau * src + (1.0 - tau) * tgt,
        nnx.state(critics.qf2),
        nnx.state(qf2_target),
    )
    nnx.update(qf1_target, new_qf1_target)
    nnx.update(qf2_target, new_qf2_target)


# ── Checkpointing ─────────────────────────────────────────────────────────────


def save_checkpoint(
    run_dir: str,
    global_step: int,
    actor_state,
    critics_state,
    qf1_target_state,
    qf2_target_state,
    alpha_state,
    actor_stats,
    critic_stats,
    verbose: bool,
):
    path = Path(run_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    mgr = ocp.CheckpointManager(
        directory=str(path),
        options=ocp.CheckpointManagerOptions(
            step_prefix="ckpt",
            step_format_fixed_length=10,
        ),
    )
    mgr.save(
        global_step,
        args=ocp.args.StandardSave(
            {
                "actor_state": actor_state,
                "critics_state": critics_state,
                "qf1_target_state": qf1_target_state,
                "qf2_target_state": qf2_target_state,
                "alpha_state": alpha_state,
                "actor_stats": actor_stats,
                "critic_stats": critic_stats,
                "global_step": jnp.array(global_step),
            }
        ),
    )
    mgr.wait_until_finished()

    if verbose:
        print(f"\n  [ckpt] saved → {path}/ckpt_{global_step}")


def load_checkpoint(
    checkpoint_path: str,
    actor_state_target,
    critics_state_target,
    qf1_target_state_target,
    qf2_target_state_target,
    alpha_state_target,
    actor_stats_target,
    critic_stats_target,
):
    path = Path(checkpoint_path).resolve()
    mgr = ocp.CheckpointManager(
        directory=str(path),
        options=ocp.CheckpointManagerOptions(
            step_prefix="ckpt",
            step_format_fixed_length=10,
        ),
    )
    step = mgr.latest_step()
    restored = mgr.restore(
        step,
        args=ocp.args.StandardRestore(
            {
                "actor_state": actor_state_target,
                "critics_state": critics_state_target,
                "qf1_target_state": qf1_target_state_target,
                "qf2_target_state": qf2_target_state_target,
                "alpha_state": alpha_state_target,
                "actor_stats": actor_stats_target,
                "critic_stats": critic_stats_target,
                "global_step": jnp.array(0),
            }
        ),
    )
    return (
        int(restored["global_step"]),
        restored["actor_state"],
        restored["critics_state"],
        restored["qf1_target_state"],
        restored["qf2_target_state"],
        restored["alpha_state"],
        restored["actor_stats"],
        restored["critic_stats"],
    )


# ── Metadata ──────────────────────────────────────────────────────────────────


def _write_metadata(run_dir: Path, run_id: str, cfg: SACConfig) -> None:
    training_cfg = vars(cfg).copy()

    try:
        raw_env = registry.load(cfg.env_id, config_overrides=cfg.env_overrides or None)
        env_config = yaml.safe_load(str(raw_env._config)) or {}
    except Exception:
        env_config = {}

    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    def _ver(pkg: str) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version(pkg)
        except Exception:
            return "unknown"

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_hash,
        "packages": {
            "jax": _ver("jax"),
            "flax": _ver("flax"),
            "optax": _ver("optax"),
            "orbax-checkpoint": _ver("orbax-checkpoint"),
            "mujoco": _ver("mujoco"),
            "mujoco-playground": _ver("mujoco-playground"),
            "brax": _ver("brax"),
            "numpy": _ver("numpy"),
        },
        "training": training_cfg,
        "environment": env_config,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


# ── Training ──────────────────────────────────────────────────────────────────


def train(cfg: SACConfig, resume_from: str | None = None):
    """
    Train SAC.

    Args:
        cfg: SACConfig with all hyperparameters.
        resume_from: path to a run dir to resume from, or None.
    """
    # ── Run identity ──────────────────────────────────────────────────
    run_id = (
        f"{cfg.env_id.lower()}"
        f"-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        f"-{uuid.uuid4().hex[:8]}"
    )
    run_dir = (Path(cfg.checkpoint_dir) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(run_dir, run_id, cfg)

    if cfg.verbose:
        print(f"  Run ID:        {run_id}")
        print(f"  Run directory: {run_dir}")
        print()

    _env_overrides = cfg.env_overrides or None

    def _load():
        return registry.load(cfg.env_id, config_overrides=_env_overrides)

    # ── Domain randomization ──────────────────────────────────────────
    key = jax.random.key(cfg.seed)
    randomization_fn = None
    if cfg.use_domain_randomization:
        _dr_fn = registry.get_domain_randomizer(cfg.env_id)
        if _dr_fn is not None:
            key, _dr_key = jax.random.split(key)
            _dr_rng = jax.random.split(_dr_key, cfg.num_envs)
            randomization_fn = lambda model: _dr_fn(model, _dr_rng)

    env = wrapper.wrap_for_brax_training(
        _load(), randomization_fn=randomization_fn, episode_length=1000, action_repeat=1
    )
    eval_env = wrapper.wrap_for_brax_training(
        _load(), episode_length=1000, action_repeat=1
    )

    def batchify(x, n):
        fix = lambda v: (
            jnp.full((n,), v) if isinstance(v, jax.Array) and v.shape == () else v
        )
        return jax.tree.map(fix, x)

    key, reset_key = jax.random.split(key)
    env_state = batchify(
        env.reset(jax.random.split(reset_key, cfg.num_envs)), cfg.num_envs
    )

    actor_obs_size = get_obs(env_state.obs, cfg.policy_obs_key).shape[-1]
    critic_obs_size = get_obs(env_state.obs, cfg.value_obs_key).shape[-1]
    action_size = env.action_size

    actor_stats = running_statistics.init_state((actor_obs_size,))
    critic_stats = running_statistics.init_state((critic_obs_size,))

    # ── Networks & optimizers ─────────────────────────────────────────
    activation = ACTIVATIONS[cfg.activation]

    key, net_key = jax.random.split(key)
    rngs = nnx.Rngs(net_key)

    actor = Actor(actor_obs_size, action_size, cfg.hidden_sizes, activation, rngs)
    qf1 = QNetwork(critic_obs_size, action_size, cfg.hidden_sizes, activation, rngs)
    qf2 = QNetwork(critic_obs_size, action_size, cfg.hidden_sizes, activation, rngs)
    qf1_target = QNetwork(
        critic_obs_size, action_size, cfg.hidden_sizes, activation, rngs
    )
    qf2_target = QNetwork(
        critic_obs_size, action_size, cfg.hidden_sizes, activation, rngs
    )
    # Initialise targets as hard copies of the online critics
    nnx.update(qf1_target, nnx.state(qf1))
    nnx.update(qf2_target, nnx.state(qf2))

    critics = Critics(qf1, qf2)

    actor_optimizer = nnx.Optimizer(
        actor, optax.adam(cfg.actor_lr, eps=1e-5), wrt=nnx.Param
    )
    critic_optimizer = nnx.Optimizer(
        critics, optax.adam(cfg.critic_lr, eps=1e-5), wrt=nnx.Param
    )

    if cfg.auto_tune_alpha:
        target_entropy = -action_size * cfg.target_entropy_scale
        alpha_module = Alpha()
        alpha_optimizer = nnx.Optimizer(
            alpha_module, optax.adam(cfg.alpha_lr, eps=1e-5), wrt=nnx.Param
        )
        alpha_val = alpha_module.value
    else:
        target_entropy = None
        alpha_module = Alpha()  # dummy — not trained
        alpha_optimizer = None
        alpha_val = jnp.array(cfg.alpha)

    # ── Replay buffer ─────────────────────────────────────────────────
    buffer = ReplayBuffer(
        buffer_size=cfg.buffer_size,
        actor_obs_size=actor_obs_size,
        critic_obs_size=critic_obs_size,
        action_size=action_size,
    )
    np_rng = np.random.default_rng(cfg.seed)

    # ── Resume ────────────────────────────────────────────────────────
    global_step = 0
    if resume_from is not None:
        (
            global_step,
            actor_state,
            critics_state,
            qf1_target_state,
            qf2_target_state,
            alpha_state,
            actor_stats,
            critic_stats,
        ) = load_checkpoint(
            resume_from,
            actor_state_target=nnx.state(actor),
            critics_state_target=nnx.state(critics),
            qf1_target_state_target=nnx.state(qf1_target),
            qf2_target_state_target=nnx.state(qf2_target),
            alpha_state_target=nnx.state(alpha_module),
            actor_stats_target=actor_stats,
            critic_stats_target=critic_stats,
        )
        nnx.update(actor, actor_state)
        nnx.update(critics, critics_state)
        nnx.update(qf1_target, qf1_target_state)
        nnx.update(qf2_target, qf2_target_state)
        nnx.update(alpha_module, alpha_state)
        if cfg.auto_tune_alpha:
            alpha_val = alpha_module.value

        if cfg.verbose:
            print(f"Resuming from {resume_from}")
            print(f"  Resumed at global_step={global_step:,}")

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=run_id,
            config={**vars(cfg), "environment": env._config.to_dict()},
            resume="allow" if resume_from else False,
        )

    # ── Discover metric keys ──────────────────────────────────────────
    metric_keys = sorted(env_state.metrics.keys())
    if cfg.verbose and metric_keys:
        print(f"  Metric keys: {metric_keys}")

    # ── Eval rollout (JIT'd via graphdef, same pattern as PPO) ────────
    actor_graphdef, _ = nnx.split(actor)

    @jax.jit
    def evaluate(actor_state, actor_stats, key):
        _actor = nnx.merge(actor_graphdef, actor_state)
        key, reset_key = jax.random.split(key)
        state = batchify(
            eval_env.reset(jax.random.split(reset_key, cfg.eval_episodes)),
            cfg.eval_episodes,
        )

        init_carry = (
            state,
            jnp.zeros(cfg.eval_episodes),
            {k: jnp.zeros(cfg.eval_episodes) for k in metric_keys},
            jnp.ones(cfg.eval_episodes),
        )

        def step(carry, _):
            state, returns, partial_returns, active = carry
            norm_a = normalize_obs(
                get_obs(state.obs, cfg.policy_obs_key), actor_stats
            )
            action = _actor.mode(norm_a)
            state = eval_env.step(state, action)
            returns = returns + state.reward * active
            partial_returns = {
                k: partial_returns[k] + state.metrics[k] * active
                for k in metric_keys
            }
            active = active * (1.0 - state.done.astype(jnp.float32))
            return (state, returns, partial_returns, active), None

        (_, returns, partial_returns, _), _ = jax.lax.scan(
            step, init_carry, None, length=1000
        )
        return returns, partial_returns

    # ── JIT running-stats update ──────────────────────────────────────
    _update_stats = jax.jit(running_statistics.update)

    # ── JIT action sampling ───────────────────────────────────────────
    # Decoupled from the update loop so the actor is callable without retrace.
    actor_graphdef_for_sample, _ = nnx.split(actor)

    @jax.jit
    def _sample_action(actor_state, norm_obs, key):
        _a = nnx.merge(actor_graphdef_for_sample, actor_state)
        return _a.sample(norm_obs, key)

    # ── Initial checkpoint ────────────────────────────────────────────
    if cfg.checkpoint_frequency > 0 and resume_from is None and cfg.use_checkpointing:
        save_checkpoint(
            run_dir=str(run_dir),
            global_step=0,
            actor_state=nnx.state(actor),
            critics_state=nnx.state(critics),
            qf1_target_state=nnx.state(qf1_target),
            qf2_target_state=nnx.state(qf2_target),
            alpha_state=nnx.state(alpha_module),
            actor_stats=actor_stats,
            critic_stats=critic_stats,
            verbose=cfg.verbose,
        )

    # ── Training loop ─────────────────────────────────────────────────
    start_time = log_time = time.time()
    eval_mean = eval_std = 0.0
    eval_partial: dict[str, float] = {}
    last_metrics: dict = {}

    total_steps = cfg.total_timesteps
    step_iter = range(global_step // cfg.num_envs, total_steps // cfg.num_envs)
    pbar = (
        tqdm(step_iter, desc="Training", unit="step", dynamic_ncols=True)
        if cfg.progress_bar
        else None
    )

    for _ in pbar if pbar is not None else step_iter:
        # ── Collect one step from all envs ────────────────────────────
        raw_a = get_obs(env_state.obs, cfg.policy_obs_key)
        raw_c = get_obs(env_state.obs, cfg.value_obs_key)

        key, step_key = jax.random.split(key)
        if global_step < cfg.learning_starts:
            action = jax.random.uniform(
                step_key, (cfg.num_envs, action_size), minval=-1.0, maxval=1.0
            )
        else:
            norm_a = normalize_obs(raw_a, actor_stats)
            _, cur_actor_state = nnx.split(actor)
            action, _ = _sample_action(cur_actor_state, norm_a, step_key)

        next_state = env.step(env_state, action)
        next_raw_a = get_obs(next_state.obs, cfg.policy_obs_key)
        next_raw_c = get_obs(next_state.obs, cfg.value_obs_key)

        # Separate termination from truncation for correct Bellman backup:
        # only mask future value for true terminations, not timeouts.
        truncation = next_state.info.get(
            "truncation", jnp.zeros_like(next_state.done)
        )
        termination = next_state.done * (1.0 - truncation)

        buffer.add_batch(
            actor_obs=np.asarray(raw_a),
            critic_obs=np.asarray(raw_c),
            next_actor_obs=np.asarray(next_raw_a),
            next_critic_obs=np.asarray(next_raw_c),
            actions=np.asarray(action),
            rewards=np.asarray(next_state.reward) * cfg.reward_scaling,
            dones=np.asarray(termination),
        )

        env_state = next_state
        global_step += cfg.num_envs

        # ── Update running stats ──────────────────────────────────────
        actor_stats = _update_stats(actor_stats, raw_a)
        critic_stats = _update_stats(critic_stats, raw_c)

        # ── Gradient updates ──────────────────────────────────────────
        if global_step >= cfg.learning_starts and buffer.size >= cfg.batch_size:
            batch = buffer.sample(cfg.batch_size, np_rng)
            key, crit_key, act_key, alph_key = jax.random.split(key, 4)

            critic_metrics = update_critic(
                actor, critics, qf1_target, qf2_target, critic_optimizer,
                batch["critic_obs"],
                batch["next_actor_obs"], batch["next_critic_obs"],
                batch["actions"], batch["rewards"], batch["dones"],
                actor_stats, critic_stats, alpha_val, cfg.gamma, crit_key,
            )

            actor_metrics = update_actor(
                actor, critics, actor_optimizer,
                batch["actor_obs"], batch["critic_obs"],
                actor_stats, critic_stats, alpha_val, act_key,
            )

            if cfg.auto_tune_alpha:
                alpha_metrics = update_alpha(
                    actor, alpha_module, alpha_optimizer,
                    batch["actor_obs"], actor_stats, target_entropy, alph_key,
                )
                alpha_val = alpha_module.value
            else:
                alpha_metrics = {"alpha": alpha_val}

            soft_update(critics, qf1_target, qf2_target, cfg.tau)

            last_metrics = {**critic_metrics, **actor_metrics, **alpha_metrics}

        # ── Eval ─────────────────────────────────────────────────────
        if global_step % cfg.eval_frequency == 0:
            key, eval_key = jax.random.split(key)
            _, cur_actor_state = nnx.split(actor)
            returns, partial_returns = evaluate(
                cur_actor_state, actor_stats, eval_key
            )
            eval_mean = float(returns.mean())
            eval_std = float(returns.std())
            eval_partial = {k: float(v.mean()) for k, v in partial_returns.items()}

        # ── Checkpoint ───────────────────────────────────────────────
        if (
            cfg.checkpoint_frequency > 0
            and cfg.use_checkpointing
            and global_step % cfg.checkpoint_frequency == 0
        ):
            save_checkpoint(
                run_dir=str(run_dir),
                global_step=global_step,
                actor_state=nnx.state(actor),
                critics_state=nnx.state(critics),
                qf1_target_state=nnx.state(qf1_target),
                qf2_target_state=nnx.state(qf2_target),
                alpha_state=nnx.state(alpha_module),
                actor_stats=actor_stats,
                critic_stats=critic_stats,
                verbose=cfg.verbose,
            )

        # ── Log ──────────────────────────────────────────────────────
        if global_step % cfg.log_frequency == 0:
            elapsed = time.time() - log_time
            sps = cfg.num_envs * cfg.log_frequency / elapsed
            log_time = time.time()

            log = {
                "time/sps": sps,
                "time/walltime": time.time() - start_time,
                "time/reward_per_step": float(next_state.reward.mean()),
                "buffer/size": buffer.size,
                **{k: float(v) for k, v in last_metrics.items()},
            }

            if global_step % cfg.eval_frequency == 0:
                log["training/episode_reward"] = eval_mean
                log["training/episode_reward_std"] = eval_std
                for k, v in eval_partial.items():
                    log[f"eval/{k}"] = v

            if cfg.verbose:
                _print = tqdm.write if pbar is not None else print
                _print(f"\n  [{global_step:>12,} / {total_steps:,}]")
                for k, v in sorted(log.items()):
                    _print(f"  {k:<40} {v:.4f}")

            if pbar is not None:
                pbar.set_postfix({"sps": f"{sps:5.0f}"})

            if cfg.use_wandb:
                wandb.log(log, step=global_step)

    if cfg.verbose:
        print(f"\nDone. Total time: {time.time() - start_time:.1f}s")

    if cfg.use_wandb:
        wandb.finish()

    return actor, critics, qf1_target, qf2_target
