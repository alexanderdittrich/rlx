"""
Render a PPO checkpoint from ppo.py.

Usage:
    python render_checkpoint.py --checkpoint checkpoints/ --output rollout.mp4
    python render_checkpoint.py --checkpoint checkpoints/ --episodes 3 --stochastic
    python render_checkpoint.py --checkpoint checkpoints/ --camera tracking_camera
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import mediapy
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from mujoco_playground import registry, wrapper
from rlx import running_statistics


# ── Import model definitions from ppo.py ─────────────────────────────────────


def _import_ppo(ppo_path: str):
    spec = importlib.util.spec_from_file_location("ppo", ppo_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ppo"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Obs helpers ───────────────────────────────────────────────────────────────


def get_obs(obs: Any, key: str) -> jax.Array:
    return obs[key] if isinstance(obs, (dict, Mapping)) else obs


def normalize_obs(raw: jax.Array, stats) -> jax.Array:
    return running_statistics.normalize(raw, stats)


# ── Checkpoint loading ────────────────────────────────────────────────────────


def load_checkpoint(
    checkpoint_path, combined_state_target, actor_stats_target, critic_stats_target
):
    path = Path(checkpoint_path).resolve()
    mgr = ocp.CheckpointManager(
        directory=str(path),
        options=ocp.CheckpointManagerOptions(step_prefix="ckpt_"),
    )
    step = mgr.latest_step()
    print(f"  Loading step {step:,} from {path}")
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Render a PPO checkpoint.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory (parent dir passed to save_checkpoint).",
    )
    parser.add_argument(
        "--ppo",
        default="src/rlx/ppo.py",
        help="Path to ppo.py (default: src/rlx/ppo.py).",
    )
    parser.add_argument("--output", default="rollout.mp4")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--camera", default=None, help="Camera name. Leave unset to use env default."
    )
    # Must match training config
    parser.add_argument("--env_id", default="Go1JoystickFlatTerrain")
    parser.add_argument(
        "--actor_hidden_sizes", nargs="+", type=int, default=[512, 256, 128]
    )
    parser.add_argument(
        "--critic_hidden_sizes", nargs="+", type=int, default=[512, 256, 128]
    )
    parser.add_argument("--activation", default="swish")
    parser.add_argument("--policy_obs_key", default="state")
    parser.add_argument("--value_obs_key", default="privileged_state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_eps", type=float, default=1e-5)
    args = parser.parse_args()

    # ── Import ActorCritic from ppo.py ────────────────────────────────────
    ppo = _import_ppo(args.ppo)
    ActorCritic = ppo.ActorCritic
    ACTIVATIONS = ppo.ACTIVATIONS

    # ── Build env ─────────────────────────────────────────────────────────
    # Use wrap_for_brax_training with num_envs=1 — this is the same wrapper
    # used during training, so obs keys / structure are identical.
    # All batched outputs will have a leading dim of 1 that we index away.
    print(f"Loading environment: {args.env_id}")
    env = wrapper.wrap_for_brax_training(
        registry.load(args.env_id), episode_length=args.episode_length, action_repeat=1
    )

    def batchify(x, n=1):
        fix = lambda v: (
            jnp.full((n,), v) if isinstance(v, jax.Array) and v.shape == () else v
        )
        return jax.tree.map(fix, x)

    key = jax.random.key(args.seed)
    key, reset_key = jax.random.split(key)

    # wrap_for_brax_training.reset expects shape (num_envs, 2) keys
    dummy_state = batchify(jax.jit(env.reset)(jax.random.split(reset_key, 1)))

    # obs has leading batch dim of 1 — drop it for shape inspection
    actor_obs_size = get_obs(dummy_state.obs, args.policy_obs_key).shape[-1]
    critic_obs_size = get_obs(dummy_state.obs, args.value_obs_key).shape[-1]
    print(
        f"  actor_obs={actor_obs_size}, critic_obs={critic_obs_size}, "
        f"action_size={env.action_size}"
    )

    # ── Build model ───────────────────────────────────────────────────────
    print("Building model...")
    key, model_key = jax.random.split(key)
    model = ActorCritic(
        actor_obs_size=actor_obs_size,
        critic_obs_size=critic_obs_size,
        action_size=env.action_size,
        actor_hidden_sizes=args.actor_hidden_sizes,
        critic_hidden_sizes=args.critic_hidden_sizes,
        activation=ACTIVATIONS[args.activation],
        rngs=nnx.Rngs(model_key),
    )
    optimizer = nnx.Optimizer(
        model=model,
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=3e-4, eps=args.adam_eps),
        ),
        wrt=nnx.Param,
    )
    graphdef, combined_state = nnx.split((model, optimizer))
    actor_stats = running_statistics.init_state((actor_obs_size,))
    critic_stats = running_statistics.init_state((critic_obs_size,))

    # ── Restore checkpoint ────────────────────────────────────────────────
    print(f"Loading checkpoint from {args.checkpoint}")
    global_step, combined_state, actor_stats, critic_stats = load_checkpoint(
        args.checkpoint, combined_state, actor_stats, critic_stats
    )
    print(f"  Checkpoint step: {global_step:,}")
    model, _ = nnx.merge(graphdef, combined_state)

    # ── JIT policy + env step ─────────────────────────────────────────────
    # State has batch dim 1. Policy operates on (1, obs_size) inputs.
    @jax.jit
    def policy_step(state, key):
        norm_a = normalize_obs(get_obs(state.obs, args.policy_obs_key), actor_stats)
        norm_c = normalize_obs(get_obs(state.obs, args.value_obs_key), critic_stats)
        dist, value = model(norm_a, norm_c)
        if args.stochastic:
            key, action_key = jax.random.split(key)
            raw = dist.sample_raw(seed=action_key)
            action = dist.postprocess(raw)
        else:
            action = dist.mode  # (1, action_size)
        next_state = env.step(state, action)
        return next_state, key, value

    # ── Render loop ───────────────────────────────────────────────────────
    action_label = "stochastic" if args.stochastic else "deterministic"
    print(f"\nRendering {args.episodes} episode(s) [{action_label}] → {args.output}")

    render_kwargs = dict(width=args.width, height=args.height)
    if args.camera is not None:
        render_kwargs["camera"] = args.camera

    all_frames = []

    for ep in range(args.episodes):
        key, ep_key = jax.random.split(key)
        state = batchify(jax.jit(env.reset)(jax.random.split(ep_key, 1)))

        ep_reward = 0.0
        ep_states = []

        for step_i in range(args.episode_length):
            # Unbatch: remove leading dim of 1, collect state before stepping
            ep_states.append(jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, state))

            state, key, _ = policy_step(state, key)
            ep_reward += float(state.reward[0])

            if bool(state.done[0]):
                break

        # env.render takes a list of State objects and returns one frame per state
        ep_frames = [np.array(f) for f in env.render(ep_states, **render_kwargs)]
        all_frames.extend(ep_frames)
        print(
            f"  Episode {ep + 1}: {len(ep_frames)} steps, "
            f"total reward = {ep_reward:.2f}"
        )

    # ── Write video ───────────────────────────────────────────────────────
    print(f"\nWriting {len(all_frames)} frames @ {args.fps} fps → {args.output}")
    mediapy.write_video(args.output, all_frames, fps=args.fps)
    print(f"Done → {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
