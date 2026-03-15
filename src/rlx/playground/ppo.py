"""
PPO for MuJoCo Playground with JAX Flax NNX and Optax.

Portions of this file are adapted from Brax
(https://github.com/google/brax),
Copyright 2022 The Brax Authors,
licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0).

Changes:
Ported to Flax NNX; single-file-oriented code structure,
hydra-config management, obs normalization warm-up.
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
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
import yaml
from flax import nnx
from mujoco_playground import registry, wrapper
from tqdm import tqdm

from rlx.common import running_statistics

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    env_id: str = "Go1JoystickFlatTerrain"
    num_envs: int = 8192

    total_timesteps: int = 200_000_000
    num_steps: int = 20  # unroll_length
    num_minibatches: int = 32  # matches Brax default
    update_epochs: int = 4  # num_updates_per_batch

    learning_rate: float = 3e-4
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    reward_scaling: float = 1.0
    norm_adv: bool = True

    actor_hidden_sizes: list[int] | None = None
    critic_hidden_sizes: list[int] | None = None
    activation: str = "swish"

    policy_obs_key: str = "state"
    value_obs_key: str = "privileged_state"

    warmup_stats: bool = True  # prime running stats before gradient updates
    log_frequency: int = 10  # log every N iterations
    eval_frequency: int = 50  # eval every N iterations
    eval_episodes: int = 128
    checkpoint_frequency: int = 100  # save checkpoint every N iterations (0=off)
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    use_checkpointing: bool = False
    wandb_project: str = "mjxrl_ppo"
    verbose: bool = True
    progress_bar: bool = False
    use_domain_randomization: bool = False
    env_overrides: dict = field(default_factory=dict)

    seed: int = 42

    def __post_init__(self):
        if self.actor_hidden_sizes is None:
            self.actor_hidden_sizes = [512, 256, 128]
        if self.critic_hidden_sizes is None:
            self.critic_hidden_sizes = [512, 256, 128]
        self.rollout_buffer_size = self.num_envs * self.num_steps
        self.batch_size = self.rollout_buffer_size // self.num_minibatches
        self.num_iterations = self.total_timesteps // self.rollout_buffer_size


# ── NormalTanhDistribution ────────────────────────────────────────────────────


class NormalTanhDistribution:
    MIN_STD: float = 0.001

    def __init__(self, params: jax.Array):
        loc, raw_scale = jnp.split(params, 2, axis=-1)
        self.loc = loc
        self.scale = jax.nn.softplus(raw_scale) + self.MIN_STD

    def _gaussian_log_prob(self, x):
        log_unnorm = -0.5 * jnp.square((x - self.loc) / self.scale)
        log_norm = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        return log_unnorm - log_norm

    @staticmethod
    def _tanh_log_jacobian(x):
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    def sample_raw(self, seed):
        return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

    def postprocess(self, raw):
        return jnp.tanh(raw)

    def log_prob(self, raw_action):
        lp = self._gaussian_log_prob(raw_action) - self._tanh_log_jacobian(raw_action)
        return jnp.sum(lp, axis=-1)

    def entropy(self, seed):
        gaussian_H = 0.5 + 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        fresh_raw = self.sample_raw(seed)
        return jnp.sum(gaussian_H + self._tanh_log_jacobian(fresh_raw), axis=-1)

    @property
    def mode(self):
        return jnp.tanh(self.loc)


# ── Network ──────────────────────────────────────────────────────────────────

ACTIVATIONS = {
    "tanh": nnx.tanh,
    "relu": nnx.relu,
    "swish": nnx.swish,
    "elu": nnx.elu,
    "gelu": nnx.gelu,
}


def _mlp(in_size, hidden_sizes, out_size, activation, rngs) -> nnx.Sequential:
    ki = nnx.initializers.lecun_uniform()
    layers, cur = [], in_size
    for h in hidden_sizes:
        layers += [nnx.Linear(cur, h, kernel_init=ki, rngs=rngs), activation]
        cur = h
    layers.append(nnx.Linear(cur, out_size, kernel_init=ki, rngs=rngs))
    return nnx.Sequential(*layers)


class ActorCritic(nnx.Module):
    def __init__(
        self,
        actor_obs_size,
        critic_obs_size,
        action_size,
        actor_hidden_sizes,
        critic_hidden_sizes,
        activation,
        rngs,
    ):
        self.policy_net = _mlp(
            actor_obs_size, actor_hidden_sizes, 2 * action_size, activation, rngs
        )
        self.value_net = _mlp(critic_obs_size, critic_hidden_sizes, 1, activation, rngs)

    def __call__(self, actor_obs, critic_obs):
        dist = NormalTanhDistribution(self.policy_net(actor_obs))
        value = self.value_net(critic_obs).squeeze(-1)
        return dist, value

    def get_value(self, critic_obs):
        return self.value_net(critic_obs).squeeze(-1)


# ── Rollout buffer ────────────────────────────────────────────────────────────


class Transition(NamedTuple):
    actor_obs: jax.Array  # raw actor obs
    critic_obs: jax.Array  # raw critic obs
    raw_action: jax.Array  # pre-tanh action
    logprob: jax.Array  # log π(raw_action) at collection time
    value: jax.Array  # V(s) at collection time
    reward: jax.Array  # scaled reward
    done: jax.Array  # episode done flag
    truncation: jax.Array  # timeout flag


# ── Obs helpers ───────────────────────────────────────────────────────────────


def get_obs(obs: Any, key: str) -> jax.Array:
    return obs[key] if isinstance(obs, (dict, Mapping)) else obs


def normalize_obs(raw: jax.Array, stats) -> jax.Array:
    return running_statistics.normalize(raw, stats)


# ── GAE — exact Brax implementation ──────────────────────────────────────────


def compute_gae(
    rewards, values, dones, truncations, bootstrap_value, gamma, gae_lambda
):
    trunc_mask = 1.0 - truncations
    termination = dones * trunc_mask

    values_next = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )
    deltas = (rewards + gamma * (1.0 - termination) * values_next - values) * trunc_mask

    def gae_step(acc, inputs):
        delta, term, tmask = inputs
        new_acc = delta + gamma * (1.0 - term) * tmask * gae_lambda * acc
        return new_acc, new_acc

    _, vs_minus_v = jax.lax.scan(
        gae_step,
        jnp.zeros_like(bootstrap_value),
        (deltas, termination, trunc_mask),
        reverse=True,
    )
    vs = vs_minus_v + values

    vs_next = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + gamma * (1.0 - termination) * vs_next - values) * trunc_mask

    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


# ── Per-minibatch update ──────────────────────────────────────────────────────


def update_minibatch(
    model,
    optimizer,
    actor_obs,
    critic_obs,
    raw_actions,
    old_logprobs,
    advantages,
    returns,
    clip_coef,
    vf_coef,
    ent_coef,
    key,
):
    def loss_fn(model: ActorCritic):
        dist, value = model(actor_obs, critic_obs)
        (key_ent,) = jax.random.split(key, 1)

        logprob = dist.log_prob(raw_actions)
        entropy = dist.entropy(key_ent).mean()

        ratio = jnp.exp(logprob - old_logprobs)
        pg1 = ratio * advantages
        pg2 = jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
        policy_loss = -jnp.minimum(pg1, pg2).mean()

        v_loss = 0.5 * vf_coef * jnp.mean((value - returns) ** 2)
        total_loss = policy_loss + v_loss - ent_coef * entropy
        return total_loss, {
            "loss/policy": policy_loss,
            "loss/value": v_loss,
            "loss/entropy": entropy,
            "loss/total": total_loss,
            "diagnostics/clip_frac": (jnp.abs(ratio - 1.0) > clip_coef).mean(),
        }

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(model)
    optimizer.update(model=model, grads=grads)
    return metrics


# ── Checkpointing ─────────────────────────────────────────────────────────────
#
# Uses Orbax for robust JAX-native checkpointing.
# graphdef is NOT saved — it is fully determined by PPOConfig and is
# reconstructed from scratch on resume. Only JAX pytree state is persisted.


def save_checkpoint(
    run_dir: str,
    global_step: int,
    combined_state,
    actor_stats,
    critic_stats,
    verbose: bool,
):
    """Save JAX pytree training state with Orbax (max_to_keep=3).

    Checkpoints are stored as ``{run_dir}/ckpt_{global_step}/``.
    """
    # TODO: asynchronous saving.
    path = Path(run_dir).resolve()  # Orbax requires absolute paths
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
                "combined_state": combined_state,
                "actor_stats": actor_stats,
                "critic_stats": critic_stats,
                "global_step": jnp.array(global_step),
            }
        ),
    )
    mgr.wait_until_finished()

    if verbose:
        print(f"\n  [ckpt] saved → {path}/ckpt_{global_step}")

    return str(path / f"ckpt_{global_step}")


def load_checkpoint(
    checkpoint_path: str,
    combined_state_target,
    actor_stats_target,
    critic_stats_target,
):
    """
    Restore a checkpoint. graphdef is NOT restored — rebuild from cfg as normal.

    Args:
        checkpoint_path: the run dir (parent of ``ckpt_*`` step subdirs).
        *_target: freshly-initialised pytrees used for shape inference by Orbax.

    Returns:
        (global_step, combined_state, actor_stats, critic_stats)
    """
    path = Path(checkpoint_path).resolve()  # Orbax requires absolute paths
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
                "combined_state": combined_state_target,
                "actor_stats": actor_stats_target,
                "critic_stats": critic_stats_target,
                "global_step": jnp.array(0),
            }
        ),
    )
    return (
        int(restored["global_step"]),
        restored["combined_state"],
        restored["actor_stats"],
        restored["critic_stats"],
    )


# ── Training ──────────────────────────────────────────────────────────────────


def _write_metadata(run_dir: Path, run_id: str, cfg: PPOConfig) -> None:
    """Write a metadata.yaml capturing everything needed to reproduce the run."""
    # Training config (all declared + computed fields)
    training_cfg = vars(cfg).copy()

    # Environment config from mujoco_playground registry (with overrides applied)
    try:
        raw_env = registry.load(cfg.env_id, config_overrides=cfg.env_overrides or None)
        env_config = yaml.safe_load(str(raw_env._config)) or {}
    except Exception:
        env_config = {}

    # Git commit for exact code version
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

    # Key package versions
    def _ver(pkg: str) -> str:
        try:
            import importlib.metadata

            return importlib.metadata.version(pkg)
        except Exception:
            return "unknown"

    packages = {
        "jax": _ver("jax"),
        "flax": _ver("flax"),
        "optax": _ver("optax"),
        "orbax-checkpoint": _ver("orbax-checkpoint"),
        "mujoco": _ver("mujoco"),
        "mujoco-playground": _ver("mujoco-playground"),
        "brax": _ver("brax"),
        "numpy": _ver("numpy"),
    }

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_hash,
        "packages": packages,
        "training": training_cfg,
        "environment": env_config,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


# ── Training ──────────────────────────────────────────────────────────────────


def train(
    cfg: PPOConfig,
    resume_from: str | None = None,
    env_factory=None,
    log_callback=None,
):
    """
    Train PPO.

    Args:
        cfg: PPOConfig with all hyperparameters.
        resume_from: path to a run dir to resume from, or None.
        env_factory: optional callable returning a raw MjxEnv instance.
            If provided, overrides registry.load(cfg.env_id).
        log_callback: optional callable(log_dict, global_step) called each
            log iteration.
    """
    # ── Run identity ──────────────────────────────────────────────────────
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
        if env_factory is not None:
            return env_factory()
        return registry.load(cfg.env_id, config_overrides=_env_overrides)

    # ── Domain randomization (Playground registry pattern) ────────────────
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

    actor_stats = running_statistics.init_state((actor_obs_size,))
    critic_stats = running_statistics.init_state((critic_obs_size,))

    key, model_key = jax.random.split(key)
    model = ActorCritic(
        actor_obs_size=actor_obs_size,
        critic_obs_size=critic_obs_size,
        action_size=env.action_size,
        actor_hidden_sizes=cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        activation=ACTIVATIONS[cfg.activation],
        rngs=nnx.Rngs(model_key),
    )
    optimizer = nnx.Optimizer(
        model=model,
        tx=optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
        ),
        wrt=nnx.Param,
    )

    graphdef, _ = nnx.split((model, optimizer))
    _, combined_state = nnx.split((model, optimizer))
    global_step = 0

    # ── Resume from checkpoint ────────────────────────────────────────────
    # graphdef is rebuilt from cfg above — only pytree state is restored.
    if resume_from is not None:
        global_step, combined_state, actor_stats, critic_stats = load_checkpoint(
            resume_from,
            combined_state_target=combined_state,
            actor_stats_target=actor_stats,
            critic_stats_target=critic_stats,
        )

        if cfg.verbose:
            print(f"Resuming from {resume_from}")
            print(f"  Resumed at global_step={global_step:,}")

    # ── Stats warmup (skipped on resume — stats already primed) ──────────
    # On iteration 0, running stats are mean=0, std=1.  We collect with
    # (obs-0)/1 = raw obs, then update stats to real values, then re-normalise
    # at loss time with those new stats → completely different network inputs
    # → ratio explodes → clip_frac ≈ 1 on the very first iteration.
    # Fix: prime the stats with one stats-only rollout before training so
    # collection and loss both see the same normalisation from step 0.
    if resume_from is None and cfg.warmup_stats:
        key, warmup_key = jax.random.split(key)

        @jax.jit
        def _warmup(env_state, actor_stats, critic_stats, key):
            model_w, _ = nnx.merge(graphdef, combined_state)

            def _step(carry, _):
                s, k = carry
                k, ak = jax.random.split(k)
                raw_a = get_obs(s.obs, cfg.policy_obs_key)
                raw_c = get_obs(s.obs, cfg.value_obs_key)
                # Use unnormalised obs — stats aren't valid yet
                dist, _ = model_w(raw_a, raw_c)
                action = dist.postprocess(dist.sample_raw(seed=ak))
                return (env.step(s, action), k), (raw_a, raw_c)

            _, (raw_as, raw_cs) = jax.lax.scan(
                _step, (env_state, key), None, length=cfg.num_steps
            )
            flat = lambda x: x.reshape(cfg.rollout_buffer_size, *x.shape[2:])
            return (
                running_statistics.update(actor_stats, flat(raw_as)),
                running_statistics.update(critic_stats, flat(raw_cs)),
            )

        actor_stats, critic_stats = _warmup(
            env_state, actor_stats, critic_stats, warmup_key
        )

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=run_id,
            config={**vars(cfg), "environment": env._config.to_dict()},
            resume="allow" if resume_from else False,
        )

    # ── Discover metric keys from env_state after reset ──────────────────
    metric_keys = sorted(env_state.metrics.keys())
    if cfg.verbose and metric_keys:
        print(f"  Metric keys: {metric_keys}")

    # ── Eval rollout ──────────────────────────────────────────────────────
    @jax.jit
    def evaluate(combined_state, actor_stats, critic_stats, key):
        model, _ = nnx.merge(graphdef, combined_state)
        key, reset_key = jax.random.split(key)
        state = batchify(
            eval_env.reset(jax.random.split(reset_key, cfg.eval_episodes)),
            cfg.eval_episodes,
        )

        # Initialise accumulators: total return + one per partial-reward key.
        # All shaped (eval_episodes,); we apply the `active` mask so episodes
        # that terminated early don't keep accumulating after done=True.
        init_carry = (
            state,
            jnp.zeros(cfg.eval_episodes),  # total return
            {
                k: jnp.zeros(cfg.eval_episodes)  # partial returns
                for k in metric_keys
            },
            jnp.ones(cfg.eval_episodes),  # active mask
        )

        def step(carry, _):
            state, returns, partial_returns, active = carry
            norm_a = normalize_obs(get_obs(state.obs, cfg.policy_obs_key), actor_stats)
            norm_c = normalize_obs(get_obs(state.obs, cfg.value_obs_key), critic_stats)
            dist, _ = model(norm_a, norm_c)
            state = eval_env.step(state, dist.mode)

            returns = returns + state.reward * active
            partial_returns = {
                k: partial_returns[k] + state.metrics[k] * active for k in metric_keys
            }
            active = active * (1.0 - state.done.astype(jnp.float32))
            return (state, returns, partial_returns, active), None

        (_, returns, partial_returns, _), _ = jax.lax.scan(
            step,
            init_carry,
            None,
            length=1000,
        )
        return returns, partial_returns

    # ── JIT training iteration ────────────────────────────────────────────
    @jax.jit
    def train_iteration(env_state, actor_stats, critic_stats, key, combined_state):
        model, optimizer = nnx.merge(graphdef, combined_state)
        key, rollout_key, update_key = jax.random.split(key, 3)

        # ── Collect rollout — store RAW obs ───────────────────────────────
        def collect_step(carry, _):
            env_state, step_key = carry
            step_key, action_key = jax.random.split(step_key)

            raw_a = get_obs(env_state.obs, cfg.policy_obs_key)
            raw_c = get_obs(env_state.obs, cfg.value_obs_key)

            # Normalise with current (pre-update) stats for action selection
            dist, value = model(
                normalize_obs(raw_a, actor_stats), normalize_obs(raw_c, critic_stats)
            )
            raw_action = dist.sample_raw(seed=action_key)
            action = dist.postprocess(raw_action)
            logprob = dist.log_prob(raw_action)

            next_state = env.step(env_state, action)
            truncation = next_state.info.get(
                "truncation", jnp.zeros_like(next_state.done)
            )

            return (next_state, step_key), Transition(
                actor_obs=raw_a,  # store RAW
                critic_obs=raw_c,  # store RAW
                raw_action=raw_action,
                logprob=logprob,
                value=value,
                reward=next_state.reward * cfg.reward_scaling,
                done=next_state.done,
                truncation=truncation,
            )

        (final_env_state, _), batch = jax.lax.scan(
            collect_step, (env_state, rollout_key), None, length=cfg.num_steps
        )

        # ── Update normaliser with full rollout (N*T samples) ────────────
        flat = lambda x: x.reshape(cfg.rollout_buffer_size, *x.shape[2:])
        new_actor_stats = running_statistics.update(actor_stats, flat(batch.actor_obs))
        new_critic_stats = running_statistics.update(
            critic_stats, flat(batch.critic_obs)
        )

        # ── Bootstrap with same stats used during collection ──────────────
        raw_c_final = get_obs(final_env_state.obs, cfg.value_obs_key)
        bootstrap_value = model.get_value(normalize_obs(raw_c_final, critic_stats))

        vs, advantages = compute_gae(
            batch.reward,
            batch.value,
            batch.done,
            batch.truncation,
            bootstrap_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        # ── Build dataset
        flat_adv = flat(advantages)
        # NOTE: We normalize advantages per-minibatch instead of globally.
        # if cfg.norm_adv:
        #     flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        dataset = (
            normalize_obs(flat(batch.actor_obs), actor_stats),
            normalize_obs(flat(batch.critic_obs), critic_stats),
            flat(batch.raw_action),
            flat(batch.logprob),
            flat_adv,
            flat(vs),
        )

        # ── PPO epochs ────────────────────────────────────────────────────
        _, current_state = nnx.split((model, optimizer))

        def run_epoch(carry, _):
            state, epoch_key = carry
            epoch_key, perm_key, loss_key = jax.random.split(epoch_key, 3)
            perm = jax.random.permutation(perm_key, cfg.rollout_buffer_size)
            minibatches = jax.tree.map(
                lambda x: x[perm].reshape(
                    cfg.num_minibatches, cfg.batch_size, *x.shape[1:]
                ),
                dataset,
            )

            def run_minibatch(carry, mb):
                state, mb_key = carry
                mb_key, step_key = jax.random.split(mb_key)
                m, o = nnx.merge(graphdef, state)

                # Normalize advantages per-minibatch comparable to Brax.
                a_obs, c_obs, raw_act, old_lp, adv, ret = mb
                if cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                mb = (a_obs, c_obs, raw_act, old_lp, adv, ret)

                metrics = update_minibatch(
                    m, o, *mb, cfg.clip_coef, cfg.vf_coef, cfg.ent_coef, step_key
                )
                _, new_state = nnx.split((m, o))
                return (new_state, mb_key), metrics

            (new_state, _), mb_metrics = jax.lax.scan(
                run_minibatch, (state, loss_key), minibatches
            )
            return (new_state, epoch_key), mb_metrics

        (final_state, _), epoch_metrics = jax.lax.scan(
            run_epoch, (current_state, update_key), None, length=cfg.update_epochs
        )

        return (
            final_env_state,
            new_actor_stats,
            new_critic_stats,
            final_state,
            key,
            jax.tree.map(jnp.mean, epoch_metrics),
            batch,
        )

    # ── Initial checkpoint (random-policy baseline) ───────────────────────
    if cfg.checkpoint_frequency > 0 and resume_from is None and cfg.use_checkpointing:
        save_checkpoint(
            run_dir=str(run_dir),
            global_step=0,
            combined_state=combined_state,
            actor_stats=actor_stats,
            critic_stats=critic_stats,
            verbose=cfg.verbose,
        )

    # ── Loop ──────────────────────────────────────────────────────────────
    start_iteration = global_step // cfg.rollout_buffer_size
    start_time = log_time = time.time()
    eval_mean = eval_std = 0.0
    eval_partial: dict[str, float] = {}

    iterations = range(start_iteration, cfg.num_iterations)
    pbar = (
        tqdm(iterations, desc="Training", unit="it", dynamic_ncols=True)
        if cfg.progress_bar
        else None
    )

    for iteration in pbar if pbar is not None else iterations:
        (env_state, actor_stats, critic_stats, combined_state, key, metrics, batch) = (
            train_iteration(env_state, actor_stats, critic_stats, key, combined_state)
        )
        global_step += cfg.rollout_buffer_size

        # ── Eval ──────────────────────────────────────────────────────────
        if iteration % cfg.eval_frequency == 0:
            key, eval_key = jax.random.split(key)
            returns, partial_returns = evaluate(
                combined_state, actor_stats, critic_stats, eval_key
            )
            eval_mean = float(returns.mean())
            eval_std = float(returns.std())
            # Mean across eval episodes for each partial reward component
            eval_partial = {k: float(v.mean()) for k, v in partial_returns.items()}

        # ── Checkpoint ────────────────────────────────────────────────────
        if (
            cfg.checkpoint_frequency > 0
            and cfg.use_checkpointing
            and (
                iteration == (cfg.num_iterations - 1)
                or iteration % cfg.checkpoint_frequency == 0
            )
        ):
            save_checkpoint(
                run_dir=str(run_dir),
                global_step=global_step,
                combined_state=combined_state,
                actor_stats=actor_stats,
                critic_stats=critic_stats,
                verbose=cfg.verbose,
            )

        # ── Log ───────────────────────────────────────────────────────────
        if iteration % cfg.log_frequency == 0:
            elapsed = time.time() - log_time
            sps = (
                cfg.rollout_buffer_size
                if iteration == start_iteration
                else cfg.rollout_buffer_size * cfg.log_frequency
            ) / elapsed
            log_time = time.time()

            log = {
                "time/sps": sps,
                "time/walltime": time.time() - start_time,
                "time/reward_per_step": float(batch.reward.mean()),
                **{k: float(v) for k, v in metrics.items()},
            }

            if iteration % cfg.eval_frequency == 0:
                log["training/episode_reward"] = eval_mean
                log["training/episode_reward_std"] = eval_std
                # Partial rewards logged as eval/reward_<component>
                for k, v in eval_partial.items():
                    log[f"eval/{k}"] = v

            if cfg.verbose:
                _print = tqdm.write if pbar is not None else print
                _print(f"\n  [{global_step:>12,} / {cfg.total_timesteps:,}]")
                for k, v in sorted(log.items()):
                    _print(f"  {k:<40} {v:.4f}")

            if pbar is not None:
                postfix: dict[str, str] = {"sps": f"{sps:5.0f}"}
                pbar.set_postfix(postfix)

            if log_callback is not None:
                log_callback(log, global_step)

            if cfg.use_wandb:
                wandb.log(log, step=global_step)

    if cfg.verbose:
        print(f"\nDone. Total time: {time.time() - start_time:.1f}s")

    if cfg.use_wandb:
        wandb.finish()

    model, optimizer = nnx.merge(graphdef, combined_state)
    return model, optimizer, actor_stats, critic_stats
