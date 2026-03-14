#!/usr/bin/env python3
"""
Parts of this code are adapted from:
https://github.com/Andrew-Luo1/rscope

MIT License

Copyright (c) 2025 Jing Yuan Luo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Browse PPO checkpoints in MuJoCo passive viewer.

Usage:
    python scripts/view_checkpoints.py <run_dir>
    python scripts/view_checkpoints.py <run_dir> --evals 5 --stochastic

Controls:
    ← / →  : previous / next evaluation for the current checkpoint
    ↓ / ↑  : earlier / later checkpoint
    space  : pause / resume
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml
from flax import nnx
from mujoco_playground import registry, wrapper

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlx.common import running_statistics
from rlx.playground.ppo import ActorCritic, ACTIVATIONS

os.environ["MUJOCO_GL"] = "glfw"


def _obs(obs, key):
    return obs[key] if isinstance(obs, (dict, Mapping)) else obs


def _batchify(x, n=1):
    return jax.tree.map(
        lambda v: (
            jnp.full((n,), v) if isinstance(v, jax.Array) and v.shape == () else v
        ),
        x,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "run_dir", help="Run directory (contains metadata.yaml and ckpt* dirs)"
    )
    ap.add_argument(
        "--evals", type=int, default=5, help="Rollouts per checkpoint (default: 3)"
    )
    ap.add_argument(
        "--episode_length",
        type=int,
        default=1000,
        help="Max steps per rollout (default: 500)",
    )
    ap.add_argument(
        "--stochastic", action="store_true", help="Sample actions stochastically"
    )
    ap.add_argument("--last", action="store_true", help="Sample actions stochastically")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()

    # ── training config from metadata ─────────────────────────────────────────
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    tr = meta["training"]
    env_id = tr["env_id"]
    pol_key = tr.get("policy_obs_key", "state")
    val_key = tr.get("value_obs_key", "privileged_state")
    actor_h = tr.get("actor_hidden_sizes", [512, 256, 128])
    critic_h = tr.get("critic_hidden_sizes", [512, 256, 128])
    act_name = tr.get("activation", "swish")
    env_overrides = tr.get("env_overrides") or None
    print(f"env: {env_id}  |  run: {run_dir.name}")

    # ── env + MuJoCo model ────────────────────────────────────────────────────
    env = wrapper.wrap_for_brax_training(
        registry.load(env_id, config_overrides=env_overrides),
        episode_length=args.episode_length,
        action_repeat=1,
    )
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    ctrl_dt = getattr(env, "dt", float(mj_model.opt.timestep) * 5)

    # ── observation sizes ─────────────────────────────────────────────────────
    key = jax.random.key(0)
    key, rk = jax.random.split(key)
    dummy = _batchify(jax.jit(env.reset)(jax.random.split(rk, 1)))
    actor_obs = _obs(dummy.obs, pol_key).shape[-1]
    critic_obs = _obs(dummy.obs, val_key).shape[-1]
    print(
        f"actor_obs={actor_obs}  critic_obs={critic_obs}  action_size={env.action_size}"
    )

    # ── model skeleton (weights loaded per checkpoint) ────────────────────────
    key, mk = jax.random.split(key)
    model = ActorCritic(
        actor_obs_size=actor_obs,
        critic_obs_size=critic_obs,
        action_size=env.action_size,
        actor_hidden_sizes=actor_h,
        critic_hidden_sizes=critic_h,
        activation=ACTIVATIONS[act_name],
        rngs=nnx.Rngs(mk),
    )
    optimizer = nnx.Optimizer(
        model=model,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4, eps=1e-5)),
        wrt=nnx.Param,
    )
    ckpt_gdef, c_state_t = nnx.split((model, optimizer))
    a_stats_t = running_statistics.init_state((actor_obs,))
    cr_stats_t = running_statistics.init_state((critic_obs,))

    # ── JIT policy step (compiled once, reused across all checkpoints) ────────
    @nnx.jit
    def policy_step(model, a_stats, cr_stats, state, key):
        na = running_statistics.normalize(_obs(state.obs, pol_key), a_stats)
        nc = running_statistics.normalize(_obs(state.obs, val_key), cr_stats)
        dist, _ = model(na, nc)
        if args.stochastic:
            key, ak = jax.random.split(key)
            action = dist.postprocess(dist.sample_raw(seed=ak))
        else:
            action = dist.mode
        return env.step(state, action), key

    # ── discover checkpoints ──────────────────────────────────────────────────
    ckpt_mgr = ocp.CheckpointManager(
        directory=str(run_dir),
        options=ocp.CheckpointManagerOptions(
            step_prefix="ckpt", step_format_fixed_length=10
        ),
    )
    steps = sorted(ckpt_mgr.all_steps())
    if args.last:
        steps = steps[-1:]

    if not steps:
        sys.exit("No checkpoints found.")
    print(f"Found {len(steps)} checkpoints: {steps}")

    # ── collect rollouts for every checkpoint ─────────────────────────────────
    # rollouts[i] = (global_step: int, evals: list of (qpos[T,nq], qvel[T,nv]))
    rollouts: list[tuple[int, list]] = []
    for i, step in enumerate(steps):
        print(f"\n  [{i + 1}/{len(steps)}] step {step:,}")
        restored = ckpt_mgr.restore(
            step,
            args=ocp.args.StandardRestore(
                {
                    "combined_state": c_state_t,
                    "actor_stats": a_stats_t,
                    "critic_stats": cr_stats_t,
                    "global_step": jnp.array(0),
                }
            ),
        )
        m, _ = nnx.merge(ckpt_gdef, restored["combined_state"])
        a_s = restored["actor_stats"]
        cr_s = restored["critic_stats"]

        key, rk = jax.random.split(key)
        state = _batchify(
            jax.jit(env.reset)(jax.random.split(rk, args.evals)), n=args.evals
        )
        all_qpos, all_qvel, all_done = [], [], []
        for _ in range(args.episode_length):
            all_qpos.append(np.array(state.data.qpos))
            all_qvel.append(np.array(state.data.qvel))
            all_done.append(np.array(state.done))
            if all_done[-1].all():
                break
            state, key = policy_step(m, a_s, cr_s, state, key)
        all_qpos = np.stack(all_qpos, axis=0)  # [T, evals, nq]
        all_qvel = np.stack(all_qvel, axis=0)  # [T, evals, nv]
        all_done = np.stack(all_done, axis=0)  # [T, evals]
        evals = []
        for e in range(args.evals):
            done_steps = np.where(all_done[:, e])[0]
            end = int(done_steps[0]) + 1 if len(done_steps) > 0 else len(all_qpos)
            evals.append((all_qpos[:end, e], all_qvel[:end, e]))
            print(f"    eval {e + 1}/{args.evals}: {end} steps")
        rollouts.append((int(restored["global_step"]), evals))

    # ── viewer state ──────────────────────────────────────────────────────────
    ci = [len(steps) - 1]  # start at latest checkpoint
    ei = [0]  # eval index
    fi = [0]  # frame index
    paused = [False]

    def key_callback(keycode):
        RIGHT, LEFT, DOWN, UP, SPACE = 262, 263, 264, 265, 32
        if keycode == RIGHT:
            ei[0] = (ei[0] + 1) % args.evals
            fi[0] = 0
        elif keycode == LEFT:
            ei[0] = (ei[0] - 1) % args.evals
            fi[0] = 0
        elif keycode == UP:
            ci[0] = min(ci[0] + 1, len(steps) - 1)
            ei[0] = 0
            fi[0] = 0
        elif keycode == DOWN:
            ci[0] = max(ci[0] - 1, 0)
            ei[0] = 0
            fi[0] = 0
        elif keycode == SPACE:
            paused[0] = not paused[0]

    print("\nControls: ← → eval  ↓ ↑ checkpoint  space pause\n")

    # ── passive viewer loop ───────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            if not paused[0]:
                _, evals = rollouts[ci[0]]
                qpos, qvel = evals[ei[0]]
                t = fi[0] % len(qpos)
                mj_data.qpos[:] = qpos[t]
                mj_data.qvel[:] = qvel[t]
                mujoco.mj_forward(mj_model, mj_data)
                fi[0] = t + 1
                if fi[0] >= len(qpos):
                    fi[0] = 0
            viewer.sync()

            # ── text overlay (top-left corner, like rscope) ──────────────
            gstep, ev = rollouts[ci[0]]
            nf = len(ev[ei[0]][0])
            left = f"Checkpoint\nEval\nFrame"
            right = (
                f"{ci[0] + 1}/{len(steps)}  (step {gstep:,})\n"
                f"{ei[0] + 1}/{args.evals}\n"
                f"{fi[0]}/{nf}"
            )
            if paused[0]:
                left += "\n[PAUSED]"
                right += "\n"
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    left,
                    right,
                )
            )

            time.sleep(ctrl_dt)


if __name__ == "__main__":
    main()
