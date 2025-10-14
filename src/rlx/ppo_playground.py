"""
PPO Implementation for MuJoCo Playground Environments

This implementation is designed for MuJoCo Playground (MJX) environments with:
- Full JAX/GPU acceleration using lax.scan
- Pure functional programming style
- Flax NNX for neural networks
- Readability and maintainability as priorities

Key Design Decisions:
1. All data stays on GPU (pure JAX arrays)
2. Use lax.scan for rollout collection, GAE, and minibatch updates
3. Maintain readability over micro-optimizations
4. Follow MuJoCo Playground API patterns

References:
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- Brax PPO: https://github.com/google/brax/tree/main/brax/training/agents/ppo
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

# Suppress warnings before imports
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import distrax
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx
from mujoco_playground import locomotion, dm_control_suite, wrapper
from omegaconf import DictConfig, OmegaConf

from rlx import running_statistics

# Set GPU parameters
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Timestamp format
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


# ---------------------------
# Configuration
# ---------------------------
@dataclass
class PPOConfig:
    # Environment
    env_id: str = "CheetahRun"  # MuJoCo Playground environment
    num_envs: int = 2048  # Large batch for MJX
    
    # Training
    total_timesteps: int = 10_000_000
    num_steps: int = 20  # Unroll length (shorter for MJX)
    batch_size: int = 512  # Minibatch size
    update_epochs: int = 4  # Epochs per update
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 1e-2
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float | None = None
    
    # Network architecture
    actor_hidden_sizes: list[int] | None = None
    critic_hidden_sizes: list[int] | None = None
    actor_activation: str = "swish"
    critic_activation: str = "swish"
    
    # Normalization
    norm_adv: bool = True
    norm_obs: bool = True  # Normalize observations (critical for continuous control)
    
    # Logging
    log_frequency: int = 10
    use_wandb: bool = False
    wandb_project: str = "ppo-playground"
    seed: int = 42
    
    # Checkpointing
    save_model: bool = False
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100
    keep_checkpoints: int = 3
    
    def __post_init__(self):
        if self.actor_hidden_sizes is None:
            self.actor_hidden_sizes = [128, 128, 128, 128]
        if self.critic_hidden_sizes is None:
            self.critic_hidden_sizes = [256, 256, 256, 256]
        
        # Calculate derived values
        self.rollout_buffer_size = self.num_envs * self.num_steps
        self.num_minibatches = self.rollout_buffer_size // self.batch_size
        self.num_iterations = self.total_timesteps // self.rollout_buffer_size
        
        assert self.rollout_buffer_size % self.batch_size == 0, (
            f"rollout_buffer_size ({self.rollout_buffer_size}) must be divisible by batch_size ({self.batch_size})"
        )


# ---------------------------
# Data Structures
# ---------------------------
class Transition(NamedTuple):
    """Single transition in the environment."""
    obs: jax.Array  # Raw observations (before normalization)
    action: jax.Array
    logprob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


# ---------------------------
# Networks
# ---------------------------
def _build_mlp(
    in_features: int,
    hidden_sizes: list[int],
    output_size: int,
    output_scale: float,
    activation_fn: callable,
    rngs: nnx.Rngs,
) -> nnx.Sequential:
    """Build an MLP with orthogonal initialization."""
    layers = []
    current_size = in_features
    
    for hidden_size in hidden_sizes:
        layers.append(
            nnx.Linear(
                current_size,
                hidden_size,
                kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            )
        )
        layers.append(activation_fn)
        current_size = hidden_size
    
    layers.append(
        nnx.Linear(
            current_size,
            output_size,
            kernel_init=nnx.initializers.orthogonal(scale=output_scale),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
    )
    
    return nnx.Sequential(*layers)


def get_activation_fn(activation: str) -> callable:
    """Get activation function by name."""
    activations = {
        "tanh": nnx.tanh,
        "relu": nnx.relu,
        "swish": nnx.swish,
        "elu": nnx.elu,
        "gelu": nnx.gelu,
    }
    
    activation_lower = activation.lower()
    if activation_lower not in activations:
        raise ValueError(
            f"Unknown activation: {activation}. Supported: {list(activations.keys())}"
        )
    
    return activations[activation_lower]


class ActorCritic(nnx.Module):
    """Actor-critic network for continuous control."""
    
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        actor_hidden_sizes: list[int],
        critic_hidden_sizes: list[int],
        actor_activation_fn: callable,
        critic_activation_fn: callable,
        rngs: nnx.Rngs,
    ):
        # Actor network (outputs action mean)
        self.policy_net = _build_mlp(
            obs_size,
            actor_hidden_sizes,
            action_size,
            output_scale=0.01,
            activation_fn=actor_activation_fn,
            rngs=rngs,
        )
        
        # Learnable log std for actions
        self.action_logstd = nnx.Param(jnp.zeros(action_size))
        
        # Critic network
        self.value_net = _build_mlp(
            obs_size,
            critic_hidden_sizes,
            1,
            output_scale=1.0,
            activation_fn=critic_activation_fn,
            rngs=rngs,
        )
    
    def __call__(self, obs: jax.Array) -> tuple[distrax.Distribution, jax.Array]:
        """Get action distribution and value estimate."""
        # Policy output (action mean)
        action_mean = self.policy_net(obs)
        
        # Value estimate
        value = self.value_net(obs).squeeze(-1)
        
        # Create diagonal Gaussian distribution
        log_std = jnp.broadcast_to(self.action_logstd.value, action_mean.shape)
        std = jnp.exp(log_std)
        dist = distrax.MultivariateNormalDiag(loc=action_mean, scale_diag=std)
        
        return dist, value
    
    def get_value(self, obs: jax.Array) -> jax.Array:
        """Get value estimate only."""
        return self.value_net(obs).squeeze(-1)


# ---------------------------
# Core Functions
# ---------------------------
# Action sampling and value functions
# ---------------------------
def sample_action(
    model: ActorCritic,
    obs: jax.Array,
    norm_state: running_statistics.RunningStatisticsState | None,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample action and compute log prob and value (with obs normalization).
    
    Args:
        model: Actor-critic model
        obs: Raw observations
        norm_state: Normalization state (None to skip normalization)
        key: Random key
        
    Returns:
        action, logprob, value
    """
    # Normalize observations if enabled
    if norm_state is not None:
        obs = running_statistics.normalize(obs, norm_state)
    
    dist, value = model(obs)
    action = dist.sample(seed=key)
    logprob = dist.log_prob(action)
    return action, logprob, value.squeeze(-1)


def get_value(model: ActorCritic, obs: jax.Array, norm_state: running_statistics.RunningStatisticsState | None) -> jax.Array:
    """Get value estimate (for bootstrapping, with obs normalization)."""
    # Normalize observations if enabled
    if norm_state is not None:
        obs = running_statistics.normalize(obs, norm_state)
    
    return model.get_value(obs)


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    next_done: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE advantages and returns using lax.scan.
    
    Args:
        rewards: [num_steps, num_envs]
        values: [num_steps, num_envs]
        dones: [num_steps, num_envs]
        next_value: [num_envs]
        next_done: [num_envs]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: [num_steps, num_envs]
        returns: [num_steps, num_envs]
    """
    num_steps = rewards.shape[0]
    
    def scan_fn(carry, step_data):
        next_value, next_advantage = carry
        reward, value, done = step_data
        
        # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        delta = reward + gamma * next_value * (1 - done) - value
        
        # GAE: A_t = δ_t + γ * λ * (1 - done_{t+1}) * A_{t+1}
        advantage = delta + gamma * gae_lambda * (1 - done) * next_advantage
        
        return (value, advantage), advantage
    
    # Scan backward through time
    _, advantages = jax.lax.scan(
        scan_fn,
        (next_value, jnp.zeros_like(next_value)),
        (rewards, values, dones),
        reverse=True,
    )
    
    # Returns are advantages + values
    returns = advantages + values
    
    return advantages, returns


def train_step(
    model: ActorCritic,
    optimizer: nnx.Optimizer,
    obs: jax.Array,  # Normalized observations
    actions: jax.Array,
    old_logprobs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    old_values: jax.Array,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    clip_vloss: bool,
) -> dict[str, jax.Array]:
    """Single training step (one minibatch, observations should already be normalized)."""
    
    def loss_fn(model: ActorCritic):
        # Get current distribution and value (obs already normalized)
        dist, value = model(obs)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Policy loss (PPO clipped objective)
        ratio = jnp.exp(logprob - old_logprobs)
        clipped_ratio = jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        policy_loss = -jnp.minimum(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean()
        
        # Value loss
        if clip_vloss:
            value_clipped = old_values + jnp.clip(
                value - old_values, -clip_coef, clip_coef
            )
            value_loss_unclipped = (value - returns) ** 2
            value_loss_clipped = (value_clipped - returns) ** 2
            value_loss = 0.5 * jnp.maximum(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((value - returns) ** 2).mean()
        
        # Total loss
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        
        return loss, {
            "loss/policy": policy_loss,
            "loss/value": value_loss,
            "loss/entropy": entropy,
            "loss/total": loss,
            "loss/approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
            "loss/clipfrac": (jnp.abs(ratio - 1) > clip_coef).astype(jnp.float32).mean(),
        }
    
    # Compute gradients
    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    
    # Update parameters (Flax 0.11.0+ requires model and grads)
    optimizer.update(model, grads)
    
    return info


# ---------------------------
# Training Loop
# ---------------------------
def train(cfg: PPOConfig):
    """Main training function."""
    
    # Load environment
    if cfg.env_id in ["CheetahRun", "HopperHop", "WalkerWalk"]:
        # DM Control Suite environment
        env = dm_control_suite.load(cfg.env_id)
    else:
        # Locomotion environment
        try:
            env = locomotion.load(cfg.env_id)
        except Exception as e:
            print(f"Failed to load locomotion environment {cfg.env_id}: {e}")
            print("Falling back to CheetahRun")
            cfg.env_id = "CheetahRun"
            env = dm_control_suite.load(cfg.env_id)
    
    # Wrap for training (auto-reset, episode tracking)
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=1000,
        action_repeat=1,
    )
    
    # Get environment info
    obs_size = env.observation_size
    action_size = env.action_size
    
    print(f"Environment: {cfg.env_id}")
    print(f"Observation size: {obs_size}")
    print(f"Action size: {action_size}")
    print(f"Number of environments: {cfg.num_envs}")
    
    # Initialize RNG
    key = jax.random.PRNGKey(cfg.seed)
    key, model_key = jax.random.split(key)
    
    # Initialize model
    actor_activation = get_activation_fn(cfg.actor_activation)
    critic_activation = get_activation_fn(cfg.critic_activation)
    
    model = ActorCritic(
        obs_size=obs_size,
        action_size=action_size,
        actor_hidden_sizes=cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        actor_activation_fn=actor_activation,
        critic_activation_fn=critic_activation,
        rngs=nnx.Rngs(model_key),
    )
    
    # Initialize optimizer
    if cfg.anneal_lr:
        lr_schedule = optax.linear_schedule(
            init_value=cfg.learning_rate,
            end_value=0.0,
            transition_steps=cfg.num_iterations,
        )
    else:
        lr_schedule = optax.constant_schedule(cfg.learning_rate)
    
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        ),
        wrt=nnx.Param,
    )
    
    # Initialize wandb
    if cfg.use_wandb:
        time_stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        run_id = f"{time_stamp}-{cfg.env_id.lower()}-ppo-{cfg.seed}"
        wandb.init(
            project=cfg.wandb_project,
            config=vars(cfg),
            name=run_id,
        )
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    # Initialize environment state
    # The wrapped environment expects a batch of keys [num_envs]
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, cfg.num_envs)
    env_state = jax.jit(env.reset)(reset_keys)
    
    global_step = 0
    
    # Define rollout collection function
    def collect_rollout(model_state, env_state, key):
        """Collect a rollout using lax.scan."""
        # Split model into graphdef and state for functional programming
        graphdef, model_params = nnx.split(model)
        
        def step_fn(carry, _):
            model_params, env_state, key = carry
            key, action_key = jax.random.split(key)
            
            # Reconstruct model from graphdef and params
            model_local = nnx.merge(graphdef, model_params)
            
            # Sample action
            action, logprob, value = sample_action(model_local, env_state.obs, action_key)
            
            # Step environment (wrapper already handles vectorization)
            next_env_state = env.step(env_state, action)
            
            # Create transition
            transition = Transition(
                obs=env_state.obs,
                action=action,
                logprob=logprob,
                value=value,
                reward=next_env_state.reward,
                done=next_env_state.done,
            )
            
            return (model_params, next_env_state, key), transition
        
        # Scan over num_steps
        (model_params, final_env_state, _), transitions = jax.lax.scan(
            step_fn,
            (model_params, env_state, key),
            None,
            length=cfg.num_steps,
        )
        
        return final_env_state, transitions
    
    # JIT compile rollout collection
    collect_rollout_jit = jax.jit(collect_rollout)
    
    # Define update function
    def update_epoch(model_params, optimizer_state, transitions, next_value, update_key):
        """Run PPO update for one epoch."""
        # Split optimizer for functional programming
        graphdef_model, _ = nnx.split(model)
        graphdef_opt, _ = nnx.split(optimizer)
        
        # Flatten batch dimension: [num_steps, num_envs, ...] -> [batch_size, ...]
        flatten = lambda x: x.reshape(cfg.rollout_buffer_size, *x.shape[2:])
        
        obs = flatten(transitions.obs)
        actions = flatten(transitions.action)
        logprobs = flatten(transitions.logprob)
        values = flatten(transitions.value)
        rewards = flatten(transitions.reward)
        dones = flatten(transitions.done)
        
        # Compute advantages and returns
        advantages, returns = compute_gae(
            transitions.reward,
            transitions.value,
            transitions.done,
            next_value,
            jnp.zeros_like(next_value),  # Assume not done after rollout
            cfg.gamma,
            cfg.gae_lambda,
        )
        
        advantages = flatten(advantages)
        returns = flatten(returns)
        
        # Normalize advantages (per minibatch during training)
        def minibatch_step(carry, indices):
            model_params, optimizer_state = carry
            
            mb_obs = obs[indices]
            mb_actions = actions[indices]
            mb_logprobs = logprobs[indices]
            mb_returns = returns[indices]
            mb_values = values[indices]
            mb_advantages = advantages[indices]
            
            # Normalize advantages per minibatch
            if cfg.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # Reconstruct model and optimizer
            model_local = nnx.merge(graphdef_model, model_params)
            optimizer_local = nnx.merge(graphdef_opt, optimizer_state)
            
            # Training step
            info = train_step(
                model_local,
                optimizer_local,
                mb_obs,
                mb_actions,
                mb_logprobs,
                mb_advantages,
                mb_returns,
                mb_values,
                cfg.clip_coef,
                cfg.vf_coef,
                cfg.ent_coef,
                cfg.clip_vloss,
            )
            
            # Split back to get updated params
            _, new_model_params = nnx.split(model_local)
            _, new_optimizer_state = nnx.split(optimizer_local)
            
            return (new_model_params, new_optimizer_state), info
        
        # Shuffle indices
        perm = jax.random.permutation(update_key, cfg.rollout_buffer_size)
        
        # Reshape into minibatches
        minibatch_indices = perm.reshape(cfg.num_minibatches, cfg.batch_size)
        
        # Scan over minibatches
        (final_model_params, final_optimizer_state), metrics = jax.lax.scan(
            minibatch_step, 
            (model_params, optimizer_state), 
            minibatch_indices
        )
        
        return final_model_params, final_optimizer_state, metrics
    
    # JIT compile update function  
    update_epoch_jit = jax.jit(update_epoch)
    
    # Split model and optimizer for functional programming
    graphdef_model, model_params = nnx.split(model)
    graphdef_opt, optimizer_state = nnx.split(optimizer)
    
    # Main training loop
    for iteration in range(cfg.num_iterations):
        iter_start_time = time.time()
        
        # Collect rollout
        key, rollout_key = jax.random.split(key)
        env_state, transitions = collect_rollout_jit(model_params, env_state, rollout_key)
        
        # Get bootstrap value
        model_temp = nnx.merge(graphdef_model, model_params)
        next_value = get_value(model_temp, env_state.obs)
        
        # Run multiple update epochs
        update_metrics = {}
        for epoch in range(cfg.update_epochs):
            key, epoch_key = jax.random.split(key)
            model_params, optimizer_state, metrics = update_epoch_jit(
                model_params, optimizer_state, transitions, next_value, epoch_key
            )
            
            # Accumulate metrics (keep as JAX arrays)
            for k, v in metrics.items():
                if k not in update_metrics:
                    update_metrics[k] = v
                else:
                    update_metrics[k] = update_metrics[k] + v
            
            # Early stopping based on KL divergence
            if cfg.target_kl is not None:
                avg_kl = float(update_metrics["loss/approx_kl"].mean()) / (epoch + 1)
                if avg_kl > cfg.target_kl:
                    print(f"Early stopping at epoch {epoch} due to KL divergence: {avg_kl:.4f}")
                    break
        
        # Update actual model and optimizer with new params
        model = nnx.merge(graphdef_model, model_params)
        optimizer = nnx.merge(graphdef_opt, optimizer_state)
        
        # Average metrics over epochs
        num_epochs_completed = epoch + 1 if cfg.target_kl is not None else cfg.update_epochs
        for k in update_metrics:
            update_metrics[k] = float(update_metrics[k].mean() / num_epochs_completed)
        
        # Update global step
        global_step += cfg.rollout_buffer_size
        
        # Calculate SPS
        iter_time = time.time() - iter_start_time
        sps = cfg.rollout_buffer_size / iter_time
        
        # Calculate episode metrics from transitions
        # Mean reward per step across all environments
        mean_reward = float(transitions.reward.mean())
        # Estimate episode return (mean reward * typical episode length)
        episode_return_estimate = mean_reward * 1000  # Approximate for 1000 step episodes
        
        # Logging
        if iteration % cfg.log_frequency == 0:
            elapsed_time = time.time() - start_time
            print(f"\nIteration {iteration}/{cfg.num_iterations}")
            print(f"  Global step: {global_step}/{cfg.total_timesteps}")
            print(f"  SPS: {sps:.0f}")
            print(f"  Elapsed: {elapsed_time:.1f}s")
            print(f"  Mean reward/step: {mean_reward:.4f}")
            print(f"  Est. episode return: {episode_return_estimate:.1f}")
            print(f"  Policy loss: {update_metrics['loss/policy']:.4f}")
            print(f"  Value loss: {update_metrics['loss/value']:.4f}")
            print(f"  Entropy: {update_metrics['loss/entropy']:.4f}")
            print(f"  Approx KL: {update_metrics['loss/approx_kl']:.4f}")
            print(f"  Clip frac: {update_metrics['loss/clipfrac']:.4f}")
            
            if cfg.use_wandb:
                wandb.log({
                    "charts/SPS": sps,
                    "charts/global_step": global_step,
                    "charts/mean_reward": mean_reward,
                    "charts/episode_return_estimate": episode_return_estimate,
                    **update_metrics,
                }, step=global_step)
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Average SPS: {cfg.total_timesteps / total_time:.0f}")
    
    return model, optimizer


# ---------------------------
# Huzzah banner
# ---------------------------
def huzzah(cfg):
    print()
    print("                     0000000                                                    ")
    print("                     0000000                       0000                         ")
    print("           000    0000 0000                        0000                         ")
    print("        00000000000      00             000 00000  0000   0000     0000         ")
    print("        0000000000        00            00000      0000     000  0000           ")
    print("        0000000000000   0000000000      000        0000      000000             ")
    print("          00000      00000000000000     000        0000      000000             ")
    print("                      00000000000000    000        0000     000  0000           ")
    print("                      0000000000000     000        0000   0000     0000         ")
    print("                        0000000000                                              ")
    print()
    print("oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
    print()
    print(f"Environment: \t\t{cfg.env_id}")
    print(f"Algorithm: \t\tPPO")
    print(f"Random Seed: \t\t{cfg.seed}")
    print(f"# envs: \t\t{cfg.num_envs}")
    print(f"# timesteps: \t\t{cfg.total_timesteps}")
    print(f"Logging directory: \t{cfg.checkpoint_dir}")
    print()


# ---------------------------
# Hydra Entry Point
# ---------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="ppo_playground")
def main(cfg: DictConfig):
    """Main entry point with Hydra config."""
    # Convert OmegaConf to dataclass
    ppo_cfg = PPOConfig(**OmegaConf.to_container(cfg, resolve=True))
    huzzah(ppo_cfg)
    train(ppo_cfg)


if __name__ == "__main__":
    main()
