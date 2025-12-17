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

from rlx.common import running_statistics

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
    num_steps: int = 20  # Unroll length - Brax uses 10-20 for locomotion
    batch_size: int = 1024  # Minibatch size
    update_epochs: int = 4  # Epochs per update

    # PPO hyperparameters
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.99  # Brax default
    gae_lambda: float = 0.95
    clip_coef: float = 0.3  # Brax default
    clip_vloss: bool = True
    ent_coef: float = 1e-2  # Brax default
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float | None = None

    # Network architecture
    actor_hidden_sizes: list[int] | None = None
    critic_hidden_sizes: list[int] | None = None
    actor_activation: str = "swish"
    critic_activation: str = "swish"
    
    # Action distribution
    action_distribution: str = "tanh_normal"  # "normal" or "tanh_normal"

    # Normalization & Scaling
    norm_adv: bool = True
    norm_obs: bool = False  # Disabled for now - adds compilation overhead
    reward_scaling: float = 1.0  # No scaling - use raw rewards

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
            self.actor_hidden_sizes = [256, 256]  # Brax default
        if self.critic_hidden_sizes is None:
            self.critic_hidden_sizes = [256, 256]  # Brax default

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
    truncation: jax.Array  # 1 if episode truncated (time limit), 0 otherwise


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

        # Critic network (use small output scale for stable initialization)
        self.value_net = _build_mlp(
            obs_size,
            critic_hidden_sizes,
            1,
            output_scale=0.01,  # Small scale to prevent large initial value predictions
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
        log_std = jnp.broadcast_to(self.action_logstd.get_value(), action_mean.shape)
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
    return action, logprob, value  # value is already squeezed in model.__call__


def get_value(
    model: ActorCritic,
    obs: jax.Array,
    norm_state: running_statistics.RunningStatisticsState | None,
) -> jax.Array:
    """Get value estimate (for bootstrapping, with obs normalization)."""
    # Normalize observations if enabled
    if norm_state is not None:
        obs = running_statistics.normalize(obs, norm_state)

    return model.get_value(obs)


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    truncations: jax.Array,
    next_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE advantages and returns using lax.scan.
    
    CRITICAL: This matches Brax's GAE implementation which distinguishes between:
    - truncation: Episode hit time limit (bootstrap value)
    - termination: Episode truly ended (don't bootstrap)

    Args:
        rewards: [num_steps, num_envs]
        values: [num_steps, num_envs]
        dones: [num_steps, num_envs] - true episode end signal
        truncations: [num_steps, num_envs] - 1 if time limit, 0 if true end
        next_value: [num_envs] - bootstrap value
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: [num_steps, num_envs]
        returns: [num_steps, num_envs]
    """
    # Compute termination: episode ended but NOT due to time limit
    # termination = done AND NOT truncated
    terminations = dones * (1 - truncations)
    
    # Truncation mask: if truncated, don't mask (allow bootstrapping)
    truncation_mask = 1 - truncations
    
    # Compute values_t_plus_1 for TD error
    # Append bootstrap value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(next_value, 0)], axis=0
    )
    
    # TD deltas: δ_t = r_t + γ * V(s_{t+1}) * (1 - termination) - V(s_t)
    # Only mask by termination (true episode end), not truncation
    deltas = rewards + gamma * (1 - terminations) * values_t_plus_1 - values
    deltas *= truncation_mask  # Zero out if truncated

    def scan_fn(carry, step_data):
        lambda_, next_advantage = carry
        truncation_mask_t, delta_t, termination_t = step_data
        
        # GAE accumulation: A_t = δ_t + γ * λ * (1 - termination) * mask * A_{t+1}
        advantage = delta_t + gamma * (1 - termination_t) * truncation_mask_t * lambda_ * next_advantage
        
        return (lambda_, advantage), advantage

    # Scan backward through time
    _, advantages = jax.lax.scan(
        scan_fn,
        (gae_lambda, jnp.zeros_like(next_value)),
        (truncation_mask, deltas, terminations),
        reverse=True,
    )

    # Returns: v_s = A_s + V(s)
    returns = advantages + values

    return advantages, returns


# Running normalization update function
def update_normalization(mean, var, count, batch_obs):
    """Update running mean and variance with new batch of observations. 
    (Welford's online algorithm)"""
    batch_mean = batch_obs.mean(axis=0)
    batch_var = batch_obs.var(axis=0)
    batch_count = jnp.array(batch_obs.shape[0], dtype=jnp.float32)
    
    delta = batch_mean - mean
    total_count = count + batch_count
    
    new_mean = mean + delta * batch_count / total_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * count * batch_count / total_count
    new_var = M2 / total_count
    
    return new_mean, new_var, total_count

# Simple normalize function
def normalize_obs(obs, mean, var):
    return (obs - mean) / jnp.sqrt(var + 1e-8)


def train_step(
    model: ActorCritic,
    optimizer: nnx.Optimizer,
    obs: jax.Array,  # Pre-normalized observations
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
    """Single training step (one minibatch, observations pre-normalized)."""

    def loss_fn(model: ActorCritic):
        # Get current distribution and value (obs pre-normalized)
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
        def compute_clipped_v_loss():
            value_clipped = old_values + jnp.clip(
                value - old_values, -clip_coef, clip_coef
            )
            value_loss_unclipped = (value - returns) ** 2
            value_loss_clipped = (value_clipped - returns) ** 2
            return 0.5 * jnp.maximum(value_loss_unclipped, value_loss_clipped).mean()

        def compute_unclipped_v_loss():
            return 0.5 * ((value - returns) ** 2).mean()

        value_loss = jax.lax.cond(
            clip_vloss, compute_clipped_v_loss, compute_unclipped_v_loss
        )

        # Total loss
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        return loss, {
            "loss/policy": policy_loss,
            "loss/value": value_loss,
            "loss/entropy": entropy,
            "loss/total": loss,
            "loss/approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
            "loss/clipfrac": (jnp.abs(ratio - 1) > clip_coef)
            .astype(jnp.float32)
            .mean(),
        }

    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, info), grads = grad_fn(model)

    # Update parameters (modifies model in-place)
    optimizer.update(model=model, grads=grads)

    return info


# ---------------------------
# Training Loop
# ---------------------------
def train(cfg: PPOConfig):
    """Main training function."""

    # Load environment
    print("Loading environment...")
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
    print("Wrapping environment...")
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=1000,
        action_repeat=1,
    )
    print("Environment wrapped!")

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

    # Build optimizer chain - only add gradient clipping if max_grad_norm is specified
    opt_chain = []
    if cfg.max_grad_norm is not None:
        opt_chain.append(optax.clip_by_global_norm(cfg.max_grad_norm))
    opt_chain.append(optax.adam(learning_rate=lr_schedule, eps=1e-5))
    
    optimizer = nnx.Optimizer(
        model,
        optax.chain(*opt_chain),
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
    start_time = time.time()

    # Initialize environment state
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, cfg.num_envs)
    print("Resetting environment...")
    env_state = env.reset(reset_keys)  # Don't double-JIT, env.reset is already compiled
    print("Environment reset complete!")

    # Initialize observation normalization with running statistics
    initial_obs = env_state.obs  # shape: (num_envs, obs_size)
    obs_count = jnp.array(cfg.num_envs, dtype=jnp.float32)
    obs_mean = initial_obs.mean(axis=0)
    obs_var = initial_obs.var(axis=0) + 1e-8
    print(f"Observation normalization initialized: mean shape={obs_mean.shape}, var shape={obs_var.shape}")
    print(f"Initial obs stats: mean={float(initial_obs.mean()):.3f}, std={float(initial_obs.std()):.3f}, min={float(initial_obs.min()):.3f}, max={float(initial_obs.max()):.3f}")
    
    # Test a single step to see reward scale
    test_key = jax.random.PRNGKey(999)
    test_action = jax.random.normal(test_key, (cfg.num_envs, action_size))
    test_state = env.step(env_state, test_action)
    print(f"Test step rewards: mean={float(test_state.reward.mean()):.6f}, std={float(test_state.reward.std()):.6f}, max={float(test_state.reward.max()):.6f}")
    print(f"Reward range in test: [{float(test_state.reward.min()):.6f}, {float(test_state.reward.max()):.6f}]")

    global_step = 0

    # Define rollout collection function using lax.scan for GPU efficiency
    @jax.jit
    def collect_rollout(model, env_state, mean, var, key):
        """Collect a rollout using lax.scan to keep computation on GPU."""
        
        def step_fn(carry, _):
            env_state, key = carry
            key, action_key = jax.random.split(key)

            # Normalize observations and sample action
            obs_norm = normalize_obs(env_state.obs, mean, var)
            action, logprob, value = sample_action(
                model, obs_norm, None, action_key
            )

            # Step environment (wrapper already handles vectorization)
            next_env_state = env.step(env_state, action)

            # Extract truncation from state.info (Brax wrapper provides this)
            # truncation=1 when episode hit time limit (bootstrap value)
            # truncation=0 when episode truly ended (agent failed)
            truncation = next_env_state.info.get('truncation', jnp.zeros_like(next_env_state.done))
            
            # Create transition (apply reward scaling)
            transition = Transition(
                obs=env_state.obs,
                action=action,
                logprob=logprob,
                value=value,
                reward=next_env_state.reward * cfg.reward_scaling,
                done=next_env_state.done,
                truncation=truncation,
            )
            
            carry = (next_env_state, key)
            return carry, transition

        # Scan over num_steps, returning final env_state and all transitions
        (final_env_state, _), transitions = jax.lax.scan(
            step_fn, (env_state, key), None, length=cfg.num_steps
        )
        
        return final_env_state, transitions

    # Define update function - JIT with lax.scan + NNX split/merge pattern
    @jax.jit
    def update_epoch(
        model, optimizer, transitions, next_value, mean, var, update_key
    ):
        """Run PPO update for one epoch."""
        # Flatten batch dimension: [num_steps, num_envs, ...] -> [batch_size, ...]
        flatten = lambda x: x.reshape(cfg.rollout_buffer_size, *x.shape[2:])

        obs = flatten(transitions.obs)
        actions = flatten(transitions.action)
        logprobs = flatten(transitions.logprob)
        values = flatten(transitions.value)

        # Normalize observations with current statistics
        obs_normalized = normalize_obs(obs, mean, var)

        # Compute advantages and returns
        # CRITICAL: Pass truncation signal to properly bootstrap at time limits
        advantages, returns = compute_gae(
            transitions.reward,
            transitions.value,
            transitions.done,
            transitions.truncation,
            next_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        advantages = flatten(advantages)
        returns = flatten(returns)

        # Shuffle indices
        perm = jax.random.permutation(update_key, cfg.rollout_buffer_size)

        # Process each minibatch using lax.scan for GPU efficiency
        # Reshape data for scanning: [num_minibatches, batch_size, ...]
        def prepare_minibatch_data(x):
            # x has shape [rollout_buffer_size, ...]
            # Reshape to [num_minibatches, batch_size, ...]
            return x[perm].reshape(cfg.num_minibatches, cfg.batch_size, *x.shape[1:])
        
        mb_obs = prepare_minibatch_data(obs_normalized)
        mb_actions = prepare_minibatch_data(actions)
        mb_logprobs = prepare_minibatch_data(logprobs)
        mb_returns = prepare_minibatch_data(returns)
        mb_values = prepare_minibatch_data(values)
        mb_advantages = prepare_minibatch_data(advantages)
        
        # Split model and optimizer into GraphDef and State for scan
        graphdef, state = nnx.split((model, optimizer))
        
        def minibatch_step(carry_state, minibatch_data):
            """Process one minibatch with functional NNX API."""
            mb_obs, mb_actions, mb_logprobs, mb_returns, mb_values, mb_advantages = minibatch_data
            
            # Normalize advantages per minibatch
            if cfg.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Merge back to stateful objects for computation
            model, optimizer = nnx.merge(graphdef, carry_state)
            
            # Training step updates model/optimizer in-place
            info = train_step(
                model,
                optimizer,
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
            
            # Split again to extract updated state
            _, new_state = nnx.split((model, optimizer))
            
            return new_state, info

        # Scan over minibatches with functional state management
        final_state, all_metrics = jax.lax.scan(
            minibatch_step,
            state,
            (mb_obs, mb_actions, mb_logprobs, mb_returns, mb_values, mb_advantages),
        )
        
        # Merge final state back into model and optimizer
        nnx.update((model, optimizer), final_state)

        # Average metrics across minibatches
        avg_metrics = jax.tree.map(lambda x: x.mean(), all_metrics)

        return avg_metrics

    # Main training loop
    for iteration in range(cfg.num_iterations):
        iter_start_time = time.time()

        # Collect rollout with current normalization statistics
        key, rollout_key = jax.random.split(key)
        env_state, transitions = collect_rollout(
            model, env_state, obs_mean, obs_var, rollout_key
        )

        # Update observation normalization statistics with new data
        # Flatten transitions to update statistics
        flat_obs = transitions.obs.reshape(-1, obs_size)
        obs_mean, obs_var, obs_count = update_normalization(
            obs_mean, obs_var, obs_count, flat_obs
        )

        # Get bootstrap value (normalize observations)
        obs_norm = normalize_obs(env_state.obs, obs_mean, obs_var)
        next_value = get_value(model, obs_norm, None)

        # Run multiple update epochs - use Python loop to avoid nested lax.scan overhead
        all_epoch_metrics = []
        for epoch in range(cfg.update_epochs):
            key, epoch_key = jax.random.split(key)
            metrics = update_epoch(
                model, optimizer, transitions, next_value, obs_mean, obs_var, epoch_key
            )
            all_epoch_metrics.append(metrics)

        # Average metrics over epochs
        update_metrics = {}
        for k in all_epoch_metrics[0].keys():
            update_metrics[k] = float(jnp.mean(jnp.array([m[k] for m in all_epoch_metrics])))

        # Update global step
        global_step += cfg.rollout_buffer_size

        # Calculate SPS
        iter_time = time.time() - iter_start_time
        sps = cfg.rollout_buffer_size / iter_time

        # Extract episode metrics from environment state (Brax wrapper tracks these)
        # The wrapper auto-resets and accumulates episode returns in state.info
        episode_metrics = {}
        if hasattr(env_state, 'info') and 'episode_metrics' in env_state.info:
            metrics_dict = env_state.info['episode_metrics']
            if 'episode_reward' in metrics_dict:
                completed_episodes = metrics_dict['episode_reward']
                # Get mean of completed episodes (filter out zeros for incomplete)
                completed_mask = completed_episodes > 0
                if completed_mask.sum() > 0:
                    episode_metrics['episode_reward'] = float(completed_episodes[completed_mask].mean())
                    episode_metrics['num_completed'] = int(completed_mask.sum())
        
        # Calculate per-step metrics from transitions  
        mean_reward = float(transitions.reward.mean())
        max_reward = float(transitions.reward.max())
        min_reward = float(transitions.reward.min())
        std_reward = float(transitions.reward.std())
        
        # Estimate episode return from per-step rewards (rough approximation)
        # Real episode tracking would require more complex logic
        episode_return_estimate = mean_reward * 1000  # Assuming 1000 step episodes
        
        # Logging
        if iteration % cfg.log_frequency == 0:
            elapsed_time = time.time() - start_time
            print(f"\nIteration {iteration}/{cfg.num_iterations}")
            print(f"  Global step: {global_step}/{cfg.total_timesteps}")
            print(f"  SPS: {sps:.0f}")
            print(f"  Elapsed: {elapsed_time:.1f}s")
            
            # Print episode metrics if available
            if episode_metrics:
                print(f"  Episode reward (completed): {episode_metrics.get('episode_reward', 0):.1f} ({episode_metrics.get('num_completed', 0)} episodes)")
            
            print(f"  Per-step rewards - mean: {mean_reward:.6f}, std: {std_reward:.6f}, max: {max_reward:.6f}")
            print(f"  Policy loss: {update_metrics['loss/policy']:.4f}")
            print(f"  Value loss: {update_metrics['loss/value']:.4f}")
            print(f"  Entropy: {update_metrics['loss/entropy']:.4f}")
            print(f"  Approx KL: {update_metrics['loss/approx_kl']:.4f}")
            print(f"  Clip frac: {update_metrics['loss/clipfrac']:.4f}")
            
            # Diagnostic info
            print(f"  Obs - mean: {transitions.obs.mean():.4f}, std: {transitions.obs.std():.4f}")
            print(f"  Actions - mean: {transitions.action.mean():.4f}, std: {transitions.action.std():.4f}")
            print(f"  Values - mean: {transitions.value.mean():.4f}, std: {transitions.value.std():.4f}")

            if cfg.use_wandb:
                wandb.log(
                    {
                        "charts/SPS": sps,
                        "charts/global_step": global_step,
                        "charts/mean_reward": mean_reward,
                        "charts/episode_return_estimate": episode_return_estimate,
                        **update_metrics,
                    },
                    step=global_step,
                )

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Average SPS: {cfg.total_timesteps / total_time:.0f}")

    return model, optimizer


# ---------------------------
# Huzzah banner
# ---------------------------
def huzzah(cfg):
    print()
    print("               666                                     ")
    print("              66666                 22                 ")
    print("       88   999666                  22                 ")
    print("    88888888     66        2222222  22   22   222      ")
    print("    88888888     55555     222      22    222222       ")
    print("      88888  55555555555   222      22     2222        ")
    print("              5555555555   222      22   222  22       ")
    print("               555555555   222      22  222    222     ")
    print("                  555                                  ")
    print()
    print("ooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
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
@hydra.main(
    version_base=None, config_path="../../configs", config_name="ppo_playground"
)
def main(cfg: DictConfig):
    """Main entry point with Hydra config."""
    # Convert OmegaConf to dataclass
    ppo_cfg = PPOConfig(**OmegaConf.to_container(cfg, resolve=True))
    huzzah(ppo_cfg)
    train(ppo_cfg)


if __name__ == "__main__":
    main()
