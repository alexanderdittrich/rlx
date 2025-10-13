"""
PPO implementation for MuJoCo Playground (MJX) environments

This is a variant of the standard PPO implementation optimized for MJX environments
from the MuJoCo Playground, which provides GPU-accelerated physics simulation.

Key differences from standard PPO:
- Uses MJX environments instead of Gymnasium
- Leverages vmap for massive parallelization across environments
- State-based environment API (functional, not object-oriented)
- No need for vectorized wrappers - native batching support

References:
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

# Suppress warnings before imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx
from omegaconf import DictConfig, OmegaConf

# Set GPU parameters
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
jax.config.update("jax_default_matmul_precision", "highest")


# ---------------------------
# Configuration
# ---------------------------
@dataclass
class PPOMJXConfig:
    # Environment
    env_name: str = "Go1JoystickFlat"  # MJX Playground environment name
    num_envs: int = 2048  # Much larger batch size for GPU

    # Training
    total_timesteps: int = 50_000_000
    num_steps: int = 20  # Shorter rollouts for faster iteration
    num_minibatches: int = 32
    update_epochs: int = 4

    # PPO hyperparameters
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    lr_schedule_type: str = "linear"  # linear, exponential, constant
    decay_rate: float = 0.99
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Network architecture
    hidden_sizes: list[int] = None  # [256, 256] by default for complex tasks

    # Normalization
    norm_adv: bool = True

    # Logging
    log_frequency: int = 10
    use_wandb: bool = True
    seed: int = 42

    # Checkpointing
    save_model: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100
    keep_checkpoints: int = 3
    resume_from: str | None = None

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_iterations = self.total_timesteps // self.batch_size


# ---------------------------
# Networks
# ---------------------------
class ActorCritic(nnx.Module):
    """Actor-Critic network for continuous control."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_sizes: list[int],
        rngs: nnx.Rngs,
    ):
        self.action_size = action_size

        # Critic network (value function)
        critic_layers = []
        in_features = obs_size
        for hidden_size in hidden_sizes:
            critic_layers.append(
                nnx.Linear(
                    in_features,
                    hidden_size,
                    kernel_init=nnx.initializers.orthogonal(scale=np.sqrt(2)),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=rngs,
                )
            )
            critic_layers.append(nnx.tanh)
            in_features = hidden_size
        critic_layers.append(
            nnx.Linear(
                in_features,
                1,
                kernel_init=nnx.initializers.orthogonal(scale=1.0),
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            )
        )
        self.critic = nnx.Sequential(*critic_layers)

        # Actor network (policy)
        actor_layers = []
        in_features = obs_size
        for hidden_size in hidden_sizes:
            actor_layers.append(
                nnx.Linear(
                    in_features,
                    hidden_size,
                    kernel_init=nnx.initializers.orthogonal(scale=np.sqrt(2)),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=rngs,
                )
            )
            actor_layers.append(nnx.tanh)
            in_features = hidden_size
        actor_layers.append(
            nnx.Linear(
                in_features,
                action_size,
                kernel_init=nnx.initializers.orthogonal(scale=0.01),
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            )
        )
        self.actor_mean = nnx.Sequential(*actor_layers)

        # Initialize log_std as learnable parameter
        self.action_logstd = nnx.Param(jnp.zeros(action_size))

    def __call__(self, x: jax.Array) -> tuple[distrax.Distribution, jax.Array]:
        """Returns (action_dist, value)."""
        value = self.critic(x).squeeze(-1)
        mean = self.actor_mean(x)
        log_std = jnp.broadcast_to(self.action_logstd.value, mean.shape)
        std = jnp.exp(log_std)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        return dist, value

    def get_value(self, x: jax.Array) -> jax.Array:
        """Get value estimate only."""
        return self.critic(x).squeeze(-1)

    def get_action_and_value(
        self,
        x: jax.Array,
        action: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Returns: (action, log_prob, entropy, value)"""
        dist, value = self(x)

        if action is None:
            action = dist.sample(seed=key)

        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, action_log_prob, entropy, value


# ---------------------------
# GAE computation
# ---------------------------
def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation."""
    num_steps = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    lastgaelam = 0.0

    for t in reversed(range(num_steps)):
        nextnonterminal = 1.0 - dones[t]
        nextvalues = next_value if t == num_steps - 1 else values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages = advantages.at[t].set(lastgaelam)

    returns = advantages + values
    return advantages, returns


# ---------------------------
# PPO Loss
# ---------------------------
def ppo_loss(
    model: ActorCritic,
    obs: jax.Array,
    actions: jax.Array,
    old_logprobs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    clip_vloss: bool,
    old_values: jax.Array | None = None,
) -> tuple[jax.Array, dict]:
    """Compute PPO loss."""
    _, new_logprobs, entropy, new_values = model.get_action_and_value(obs, actions)

    # Policy loss
    logratio = new_logprobs - old_logprobs
    ratio = jnp.exp(logratio)
    approx_kl = ((ratio - 1) - logratio).mean()

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    v_loss_unclipped = (new_values - returns) ** 2
    v_clipped = old_values + jnp.clip(new_values - old_values, -clip_coef, clip_coef)
    v_loss_clipped = (v_clipped - returns) ** 2
    v_loss = jax.lax.cond(
        clip_vloss,
        lambda: 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean(),
        lambda: 0.5 * v_loss_unclipped.mean(),
    )

    # Entropy bonus
    entropy_loss = -entropy.mean()

    # Total loss
    loss = pg_loss + vf_coef * v_loss + ent_coef * entropy_loss

    # Diagnostics
    clipfrac = (jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32).mean()

    info = {
        "loss/total": loss,
        "loss/policy": pg_loss,
        "loss/value": v_loss,
        "loss/entropy": entropy_loss,
        "diagnostics/approx_kl": approx_kl,
        "diagnostics/clipfrac": clipfrac,
        "diagnostics/explained_variance": 1
        - jnp.var(returns - new_values) / (jnp.var(returns) + 1e-8),
    }

    return loss, info


@nnx.jit
def train_step(
    model: ActorCritic,
    optimizer: nnx.Optimizer,
    obs: jax.Array,
    actions: jax.Array,
    old_logprobs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    old_values: jax.Array,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    clip_vloss: bool,
) -> dict:
    """Single training step."""
    grad_fn = nnx.value_and_grad(ppo_loss, has_aux=True)
    (loss, info), grads = grad_fn(
        model,
        obs,
        actions,
        old_logprobs,
        advantages,
        returns,
        clip_coef,
        vf_coef,
        ent_coef,
        clip_vloss,
        old_values,
    )

    # Compute gradient norm for logging (before clipping by optimizer)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    info["diagnostics/grad_norm"] = grad_norm

    # Optimizer applies gradient clipping via optax.clip_by_global_norm
    optimizer.update(model=model, grads=grads)

    return info


# ---------------------------
# Checkpoint utilities
# ---------------------------
def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: ActorCritic,
    global_step: int,
    num_updates: int,
    key: jax.Array,
    metrics: dict,
) -> None:
    """Save a checkpoint with the current model state and training progress."""
    checkpoint_state = {
        "model_state": nnx.state(model),
        "global_step": global_step,
        "num_updates": num_updates,
        "rng_key": key,
    }

    checkpoint_manager.save(
        step=num_updates,
        args=ocp.args.StandardSave(checkpoint_state),
        metrics=metrics,
    )
    checkpoint_manager.wait_until_finished()


def load_checkpoint(
    checkpoint_path: str,
    model: ActorCritic,
) -> tuple[int, int, jax.Array]:
    """Load a checkpoint and restore model state."""
    checkpoint_manager = ocp.CheckpointManager(
        directory=str(Path(checkpoint_path).resolve()),
        options=ocp.CheckpointManagerOptions(create=False),
    )

    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError(f"No checkpoints found in {checkpoint_path}")

    target_checkpoint = {
        "model_state": nnx.state(model),
        "global_step": 0,
        "num_updates": 0,
        "rng_key": jax.random.key(0),
    }

    checkpoint_state = checkpoint_manager.restore(
        latest_step, args=ocp.args.StandardRestore(target_checkpoint)
    )

    nnx.update(model, checkpoint_state["model_state"])

    print(f"Loaded checkpoint from step {latest_step}")
    print(f"  Global step: {checkpoint_state['global_step']}")
    print(f"  Updates: {checkpoint_state['num_updates']}")

    return (
        checkpoint_state["global_step"],
        checkpoint_state["num_updates"],
        checkpoint_state["rng_key"],
    )


# ---------------------------
# Main training loop
# ---------------------------
def train(cfg: PPOMJXConfig):
    """Main PPO training loop for MJX environments."""
    # Import here to avoid dependency issues if playground not installed
    try:
        from mujoco_playground import registry
    except ImportError:
        raise ImportError(
            "mujoco_playground is required. Install with: pip install playground"
        )

    # Set random seeds
    np.random.seed(cfg.seed)
    key = jax.random.key(cfg.seed)

    # Load MJX environment
    print(f"Loading environment: {cfg.env_name}")
    env = registry.load(cfg.env_name)
    
    # Get observation and action sizes
    obs_size = env.observation_size
    action_size = env.action_size
    
    print(f"Observation size: {obs_size}")
    print(f"Action size: {action_size}")
    print(f"Number of parallel environments: {cfg.num_envs}")

    # Initialize model
    key, model_key = jax.random.split(key)
    model = ActorCritic(
        obs_size=obs_size,
        action_size=action_size,
        hidden_sizes=cfg.hidden_sizes,
        rngs=nnx.Rngs(model_key),
    )

    # Create learning rate schedule
    if cfg.anneal_lr:
        total_steps = cfg.num_iterations * cfg.update_epochs * cfg.num_minibatches

        if cfg.lr_schedule_type == "linear":
            lr_schedule = optax.linear_schedule(
                init_value=cfg.learning_rate,
                end_value=0.0,
                transition_steps=total_steps,
            )
        elif cfg.lr_schedule_type == "exponential":
            lr_schedule = optax.exponential_decay(
                init_value=cfg.learning_rate,
                transition_steps=total_steps,
                decay_rate=cfg.decay_rate,
                alpha=0.0,
            )
        elif cfg.lr_schedule_type == "constant":
            lr_schedule = optax.constant_schedule(cfg.learning_rate)
        else:
            raise ValueError(f"Unknown lr_schedule_type: {cfg.lr_schedule_type}")
    else:
        lr_schedule = optax.constant_schedule(cfg.learning_rate)

    # Initialize optimizer with gradient clipping
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=lr_schedule),
        ),
        wrt=nnx.Param,
    )

    # Setup checkpointing
    checkpoint_manager = None
    if cfg.save_model:
        checkpoint_dir = Path(cfg.checkpoint_dir).resolve() / f"{cfg.env_name}-{cfg.seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_manager = ocp.CheckpointManager(
            directory=str(checkpoint_dir),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=cfg.keep_checkpoints,
                create=True,
            ),
        )

    # Initialize wandb
    if cfg.use_wandb:
        wandb.init(
            project="ppo-mjx",
            config=vars(cfg),
            name=f"{cfg.env_name}-{cfg.seed}",
        )

    # Initialize environment states
    print("Initializing environment states...")
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, cfg.num_envs)
    
    # JIT compile the reset and step functions
    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step = jax.jit(jax.vmap(env.step))
    
    env_state = jit_reset(reset_keys)
    
    global_step = 0
    num_updates = 0

    # Resume from checkpoint if specified
    if cfg.resume_from is not None:
        global_step, num_updates, key = load_checkpoint(cfg.resume_from, model)

    # Storage for rollouts
    rollout_obs = []
    rollout_actions = []
    rollout_logprobs = []
    rollout_rewards = []
    rollout_dones = []
    rollout_values = []

    # Episode tracking
    episode_returns = []
    episode_lengths = []

    start_time = time.time()

    print("Starting training...")
    for iteration in range(cfg.num_iterations):
        # Collect rollout
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs

            # Sample actions
            key, action_key = jax.random.split(key)
            action_keys = jax.random.split(action_key, cfg.num_envs)
            
            obs = env_state.obs
            action, logprob, _, value = jax.vmap(
                lambda o, k: model.get_action_and_value(o, key=k)
            )(obs, action_keys)

            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_logprobs.append(logprob)
            rollout_values.append(value)

            # Step environment
            env_state = jit_step(env_state, action)

            rollout_rewards.append(env_state.reward)
            rollout_dones.append(env_state.done)

            # Log episode stats
            done_mask = env_state.done
            if done_mask.any():
                done_indices = jnp.where(done_mask)[0]
                for idx in done_indices:
                    # Try to get episode return from metrics
                    if hasattr(env_state, 'metrics') and 'episode_return' in env_state.metrics:
                        episode_returns.append(float(env_state.metrics['episode_return'][idx]))
                    elif hasattr(env_state, 'info') and 'episode_return' in env_state.info:
                        episode_returns.append(float(env_state.info['episode_return'][idx]))

        # Stack rollout data
        rollout_obs = jnp.stack(rollout_obs)
        rollout_actions = jnp.stack(rollout_actions)
        rollout_logprobs = jnp.stack(rollout_logprobs)
        rollout_rewards = jnp.stack(rollout_rewards)
        rollout_dones = jnp.stack(rollout_dones)
        rollout_values = jnp.stack(rollout_values)

        # Bootstrap value for GAE
        next_value = jax.vmap(model.get_value)(env_state.obs)

        # Compute advantages and returns
        advantages, returns = compute_gae(
            rollout_rewards,
            rollout_values,
            rollout_dones,
            next_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        # Flatten batch
        b_obs = rollout_obs.reshape((-1, obs_size))
        b_actions = rollout_actions.reshape((-1, action_size))
        b_logprobs = rollout_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = rollout_values.reshape(-1)

        # Normalize advantages
        if cfg.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

        # Clear rollout storage
        rollout_obs = []
        rollout_actions = []
        rollout_logprobs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        # Update policy
        update_info = {}
        for epoch in range(cfg.update_epochs):
            # Random permutation for minibatches
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, cfg.batch_size)

            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = perm[start:end]

                info = train_step(
                    model=model,
                    optimizer=optimizer,
                    obs=b_obs[mb_inds],
                    actions=b_actions[mb_inds],
                    old_logprobs=b_logprobs[mb_inds],
                    advantages=b_advantages[mb_inds],
                    returns=b_returns[mb_inds],
                    old_values=b_values[mb_inds],
                    clip_coef=cfg.clip_coef,
                    vf_coef=cfg.vf_coef,
                    ent_coef=cfg.ent_coef,
                    clip_vloss=cfg.clip_vloss,
                )

                # Accumulate info
                for k, v in info.items():
                    update_info[k] = update_info.get(k, 0) + float(v)

            # Check KL divergence for early stopping
            if cfg.target_kl is not None:
                avg_kl = update_info.get("diagnostics/approx_kl", 0) / (
                    (epoch + 1) * cfg.num_minibatches
                )
                if avg_kl > 1.5 * cfg.target_kl:
                    print(f"Early stopping at epoch {epoch} due to KL={avg_kl:.4f}")
                    break

        num_updates += 1

        # Average update info
        num_minibatch_updates = (epoch + 1) * cfg.num_minibatches
        for k in update_info:
            update_info[k] /= num_minibatch_updates

        # Logging
        if num_updates % cfg.log_frequency == 0:
            sps = int(global_step / (time.time() - start_time))

            # Get current learning rate from scheduler
            current_step = num_updates * cfg.update_epochs * cfg.num_minibatches
            current_lr = lr_schedule(current_step)

            log_dict = {
                "global_step": global_step,
                "num_updates": num_updates,
                "sps": sps,
                "learning_rate": float(current_lr),
                **update_info,
            }

            if episode_returns:
                log_dict["episode/return_mean"] = np.mean(episode_returns)
                log_dict["episode/return_std"] = np.std(episode_returns)
                last_episode_return = log_dict["episode/return_mean"]
                episode_returns = []

            if cfg.use_wandb:
                wandb.log(log_dict, step=global_step)

            print(
                f"Step {global_step}: SPS={sps}, Loss={update_info['loss/total']:.4f}"
            )
            if "episode/return_mean" in log_dict:
                print(
                    f"  Episode Return: {log_dict['episode/return_mean']:.2f} ± {log_dict['episode/return_std']:.2f}"
                )

        # Save checkpoint periodically
        if (
            cfg.save_model
            and checkpoint_manager is not None
            and num_updates % cfg.checkpoint_frequency == 0
        ):
            episode_return_metric = 0.0
            if episode_returns:
                episode_return_metric = float(np.mean(episode_returns))
            elif 'last_episode_return' in locals():
                episode_return_metric = float(last_episode_return)

            save_checkpoint(
                checkpoint_manager=checkpoint_manager,
                model=model,
                global_step=global_step,
                num_updates=num_updates,
                key=key,
                metrics={
                    "global_step": global_step,
                    "episode_return": episode_return_metric,
                },
            )
            print(f"Saved checkpoint at update {num_updates}")

    # Save final checkpoint
    if cfg.save_model and checkpoint_manager is not None:
        save_checkpoint(
            checkpoint_manager=checkpoint_manager,
            model=model,
            global_step=global_step,
            num_updates=num_updates,
            key=key,
            metrics={"global_step": global_step, "final": True},
        )
        print(f"Saved final checkpoint at update {num_updates}")
        checkpoint_manager.close()

    if cfg.use_wandb:
        wandb.finish()

    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    print(f"Final SPS: {int(global_step / (time.time() - start_time))}")


# ---------------------------
# Hydra entry point
# ---------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="ppo_mjx")
def main(cfg: DictConfig):
    """Main entry point."""
    ppo_cfg = PPOMJXConfig(**OmegaConf.to_container(cfg, resolve=True))
    train(ppo_cfg)


if __name__ == "__main__":
    main()
