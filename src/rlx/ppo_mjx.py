"""
PPO implementation for MuJoCo Playground (MJX) environments

This is a variant of the standard PPO implementation optimized for MJX environments
from the MuJoCo Playground, which provides GPU-accelerated physics simulation.

Key differences from standard PPO:
- Uses MJX environments instead of Gymnasium
- Leverages vmap for massive parallelization across environments
- State-based environment API (functional, not object-oriented)
- No need for vectorized wrappers - native batching support

JAX Optimizations:
- Uses jax.lax.scan for rollout collection (fully JIT-compiled)
- Uses jax.lax.scan for GAE computation (backward pass)
- Uses jax.lax.scan for minibatch updates (epoch training)
- All major computations are JIT-compiled for maximum performance
- Eliminates Python loops for data collection and training updates

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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
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

# Timestamp format for run IDs
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


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
    num_steps: int = 20  # rollout length per env (shorter for MJX)
    batch_size: int = 64  # minibatch size for training (same as SB3 default)
    update_epochs: int = 4  # n_epochs in SB3

    # PPO hyperparameters
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    lr_schedule_type: str = "linear"  # linear, exponential, constant
    decay_rate: float = 0.99
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0  # SB3 default (use 0.01 for more exploration)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Network architecture
    actor_hidden_sizes: list[int] | None = None  # Actor/policy network hidden layers
    critic_hidden_sizes: list[int] | None = None  # Critic/value network hidden layers
    # Actor/policy network activation: tanh, relu, swish, elu, gelu
    actor_activation: str = "tanh"
    # Critic/value network activation: tanh, relu, swish, elu, gelu
    critic_activation: str = "tanh"

    # Normalization
    norm_adv: bool = True

    # Logging
    log_frequency: int = 10
    use_wandb: bool = True
    wandb_project: str = "ppo-mjx"  # Wandb project name
    seed: int = 42

    # Checkpointing
    save_model: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100
    keep_checkpoints: int = 3
    resume_from: str | None = None

    def __post_init__(self):
        if self.actor_hidden_sizes is None:
            self.actor_hidden_sizes = [256, 256]
        if self.critic_hidden_sizes is None:
            self.critic_hidden_sizes = [256, 256]
        
        # Calculate derived values (matching SB3 naming)
        self.rollout_buffer_size = self.num_envs * self.num_steps
        self.num_minibatches = self.rollout_buffer_size // self.batch_size
        self.num_iterations = self.total_timesteps // self.rollout_buffer_size
        
        # Sanity check (matching SB3)
        assert self.rollout_buffer_size > 1 or not self.norm_adv, (
            "`num_envs * num_steps` must be greater than 1 when using advantage normalization"
        )
        assert self.batch_size > 1 or not self.norm_adv, (
            "`batch_size` must be greater than 1 when using advantage normalization"
        )


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
    """Build an MLP with orthogonal initialization (SB3 standard).

    Args:
        in_features: Input dimension
        hidden_sizes: List of hidden layer sizes
        output_size: Output dimension
        output_scale: Orthogonal initialization scale for output layer (0.01 for policy, 1.0 for value)
        activation_fn: Activation function (nnx.tanh, nnx.relu, nnx.swish, nnx.elu, etc.)
        rngs: Random number generators

    Returns:
        Sequential network
    """
    layers = []
    current_size = in_features

    # Hidden layers with activation
    for hidden_size in hidden_sizes:
        layers.append(
            nnx.Linear(
                current_size,
                hidden_size,
                kernel_init=nnx.initializers.orthogonal(scale=np.sqrt(2)),
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            )
        )
        layers.append(activation_fn)
        current_size = hidden_size

    # Output layer
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
    """Get activation function by name.

    Args:
        activation: Name of activation function (tanh, relu, swish, elu, etc.)

    Returns:
        Activation function

    Raises:
        ValueError: If activation name is not recognized
    """
    activations = {
        "tanh": nnx.tanh,
        "relu": nnx.relu,
        "swish": nnx.swish,
        "elu": nnx.elu,
        "gelu": nnx.gelu,
        "leaky_relu": nnx.leaky_relu,
    }

    if activation not in activations:
        raise ValueError(
            f"Unknown activation: {activation}. "
            f"Available: {list(activations.keys())}"
        )

    return activations[activation]


class ActorCritic(nnx.Module):
    """Actor-Critic network for continuous control."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        actor_hidden_sizes: list[int],
        critic_hidden_sizes: list[int],
        actor_activation: str = "tanh",
        critic_activation: str = "tanh",
        rngs: nnx.Rngs = None,
    ):
        self.action_size = action_size

        # Get activation functions
        actor_activation_fn = get_activation_fn(actor_activation)
        critic_activation_fn = get_activation_fn(critic_activation)

        # Build critic network (value function)
        self.critic = _build_mlp(
            in_features=obs_size,
            hidden_sizes=critic_hidden_sizes,
            output_size=1,
            output_scale=1.0,
            activation_fn=critic_activation_fn,
            rngs=rngs,
        )

        # Build actor network (policy mean)
        self.actor_mean = _build_mlp(
            in_features=obs_size,
            hidden_sizes=actor_hidden_sizes,
            output_size=action_size,
            output_scale=0.01,
            activation_fn=actor_activation_fn,
            rngs=rngs,
        )

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
# Rollout collection with scan
# ---------------------------
def collect_rollout_step(carry, unused):
    """Single step of rollout collection for jax.lax.scan.
    
    Args:
        carry: (model, env_state, key)
        unused: Unused input (for scan)
        
    Returns:
        new_carry: Updated (model, env_state, key)
        output: (obs, action, logprob, value, reward, done)
    """
    model, env_state, env_step_fn, key = carry
    
    # Get observation
    obs = env_state.obs
    
    # Sample actions (vectorized across environments)
    key, action_key = jax.random.split(key)
    action_keys = jax.random.split(action_key, obs.shape[0])
    action, logprob, _, value = jax.vmap(
        lambda o, k: model.get_action_and_value(o, key=k)
    )(obs, action_keys)
    
    # Step environment (already vmapped)
    env_state = env_step_fn(env_state, action)
    
    # Return updated carry and collected data
    new_carry = (model, env_state, env_step_fn, key)
    output = (obs, action, logprob, value, env_state.reward, env_state.done)
    
    return new_carry, output


def collect_rollout(
    model: ActorCritic,
    env_state: Any,
    env_step_fn: callable,
    key: jax.Array,
    num_steps: int,
) -> tuple[Any, dict]:
    """Collect a rollout using jax.lax.scan for efficiency.
    
    Args:
        model: The actor-critic model
        env_state: Initial environment state
        env_step_fn: JIT-compiled vmapped environment step function
        key: Random key
        num_steps: Number of steps to collect
        
    Returns:
        env_state: Final environment state
        rollout_data: Dictionary with trajectories (obs, actions, logprobs, values, rewards, dones)
    """
    # Use scan to collect rollout
    carry = (model, env_state, env_step_fn, key)
    final_carry, outputs = jax.lax.scan(
        collect_rollout_step,
        carry,
        None,
        length=num_steps
    )
    
    # Unpack outputs
    obs, actions, logprobs, values, rewards, dones = outputs
    final_env_state = final_carry[1]
    final_key = final_carry[3]
    
    return final_env_state, final_key, {
        'obs': obs,
        'actions': actions,
        'logprobs': logprobs,
        'values': values,
        'rewards': rewards,
        'dones': dones,
    }


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
    """Compute Generalized Advantage Estimation using scan for efficiency."""
    
    def gae_step(carry, transition):
        """Single GAE computation step (backward pass)."""
        gae = carry
        reward, value, done, next_value = transition
        
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        
        return gae, gae
    
    # Append next_value for the last step
    next_values = jnp.concatenate([values[1:], next_value[None]], axis=0)
    
    # Run GAE computation backward
    _, advantages = jax.lax.scan(
        gae_step,
        jnp.zeros_like(next_value),  # Initial GAE
        (rewards, values, dones, next_values),
        reverse=True
    )
    
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
        "loss/approx_kl": approx_kl,
        "loss/clipfrac": clipfrac,
        "loss/explained_variance": 1
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
    info["loss/grad_norm"] = grad_norm

    # Optimizer applies gradient clipping via optax.clip_by_global_norm
    optimizer.update(model=model, grads=grads)

    return info


def update_epoch(
    model: ActorCritic,
    optimizer: nnx.Optimizer,
    obs: jax.Array,
    actions: jax.Array,
    logprobs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    values: jax.Array,
    key: jax.Array,
    cfg: PPOMJXConfig,
) -> tuple[jax.Array, dict]:
    """Perform one epoch of minibatch updates using scan.
    
    Args:
        model: Actor-critic model
        optimizer: Optimizer
        obs: Observations [rollout_buffer_size, obs_dim]
        actions: Actions [rollout_buffer_size, action_dim]
        logprobs: Log probabilities [rollout_buffer_size]
        advantages: Advantages [rollout_buffer_size]
        returns: Returns [rollout_buffer_size]
        values: Values [rollout_buffer_size]
        key: Random key
        cfg: Configuration
        
    Returns:
        key: Updated random key
        info: Dictionary of training metrics
    """
    # Shuffle indices
    key, perm_key = jax.random.split(key)
    perm = jax.random.permutation(perm_key, cfg.rollout_buffer_size)
    
    # Shuffle all data
    obs = obs[perm]
    actions = actions[perm]
    logprobs = logprobs[perm]
    advantages = advantages[perm]
    returns = returns[perm]
    values = values[perm]
    
    # Reshape into minibatches [num_minibatches, batch_size, ...]
    def reshape_minibatches(x):
        return x.reshape(cfg.num_minibatches, cfg.batch_size, *x.shape[1:])
    
    mb_obs = reshape_minibatches(obs)
    mb_actions = reshape_minibatches(actions)
    mb_logprobs = reshape_minibatches(logprobs)
    mb_advantages = reshape_minibatches(advantages)
    mb_returns = reshape_minibatches(returns)
    mb_values = reshape_minibatches(values)
    
    def minibatch_step(carry, minibatch_data):
        """Single minibatch update."""
        model, optimizer = carry
        mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values = minibatch_data
        
        # Normalize advantages per minibatch
        if cfg.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        
        # Perform training step
        info = train_step(
            model=model,
            optimizer=optimizer,
            obs=mb_obs,
            actions=mb_actions,
            old_logprobs=mb_logprobs,
            advantages=mb_advantages,
            returns=mb_returns,
            old_values=mb_values,
            clip_coef=cfg.clip_coef,
            vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef,
            clip_vloss=cfg.clip_vloss,
        )
        
        return (model, optimizer), info
    
    # Run all minibatches using scan
    _, infos = jax.lax.scan(
        minibatch_step,
        (model, optimizer),
        (mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values)
    )
    
    # Average metrics across minibatches
    avg_info = {k: v.mean() for k, v in infos.items()}
    
    return key, avg_info


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
        actor_hidden_sizes=cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        actor_activation=cfg.actor_activation,
        critic_activation=cfg.critic_activation,
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
    # Note: Using eps=1e-5 to match CleanRL's Adam optimizer
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        ),
        wrt=nnx.Param,
    )

    # Create run ID with timestamp
    time_stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    # Environment names in MJX don't have version suffixes, but clean just in case
    env_name_clean = cfg.env_name.lower()
    run_id = f"{time_stamp}-{env_name_clean}-ppo-{cfg.seed}"

    # Setup checkpointing
    checkpoint_manager = None
    if cfg.save_model:
        checkpoint_dir = Path(cfg.checkpoint_dir).resolve() / run_id
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
            project=cfg.wandb_project,
            config=vars(cfg),
            name=run_id,
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

    # Episode tracking
    episode_returns = []
    episode_lengths = []

    start_time = time.time()

    # JIT compile the rollout collection function
    jit_collect_rollout = jax.jit(collect_rollout, static_argnums=(4,))
    
    # JIT compile GAE computation
    jit_compute_gae = jax.jit(compute_gae, static_argnums=(4, 5))

    print("Starting training...")
    for iteration in range(cfg.num_iterations):
        # Collect rollout using scan (fully JIT-compiled)
        env_state, key, rollout_data = jit_collect_rollout(
            model,
            env_state,
            jit_step,
            key,
            cfg.num_steps,
        )
        
        global_step += cfg.num_steps * cfg.num_envs
        
        # Extract rollout data
        rollout_obs = rollout_data['obs']
        rollout_actions = rollout_data['actions']
        rollout_logprobs = rollout_data['logprobs']
        rollout_values = rollout_data['values']
        rollout_rewards = rollout_data['rewards']
        rollout_dones = rollout_data['dones']
        
        # Log episode stats (extract from final environment state)
        # Note: This is approximate - we log episodes that finished during rollout
        done_mask = rollout_dones.any(axis=0)  # Any episodes that finished
        if done_mask.any():
            # Try to get episode return from metrics
            if hasattr(env_state, "metrics") and "episode_return" in env_state.metrics:
                finished_returns = env_state.metrics["episode_return"][done_mask]
                episode_returns.extend([float(r) for r in finished_returns])
            elif hasattr(env_state, "info") and "episode_return" in env_state.info:
                finished_returns = env_state.info["episode_return"][done_mask]
                episode_returns.extend([float(r) for r in finished_returns])

        # Bootstrap value for GAE
        next_value = jax.vmap(model.get_value)(env_state.obs)

        # Compute advantages and returns (JIT-compiled with scan)
        advantages, returns = jit_compute_gae(
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

        # Update policy using scan-based epochs
        update_info = {}
        for epoch in range(cfg.update_epochs):
            key, epoch_info = update_epoch(
                model=model,
                optimizer=optimizer,
                obs=b_obs,
                actions=b_actions,
                logprobs=b_logprobs,
                advantages=b_advantages,
                returns=b_returns,
                values=b_values,
                key=key,
                cfg=cfg,
            )
            
            # Accumulate info
            for k, v in epoch_info.items():
                update_info[k] = update_info.get(k, 0) + float(v)
            
            # Check KL divergence for early stopping
            if cfg.target_kl is not None:
                avg_kl = update_info["loss/approx_kl"] / (epoch + 1)
                if avg_kl > 1.5 * cfg.target_kl:
                    print(f"Early stopping at epoch {epoch} due to KL={avg_kl:.4f}")
                    break

        num_updates += 1

        # Average update info across epochs
        num_epochs_completed = epoch + 1
        for k in update_info:
            update_info[k] /= num_epochs_completed

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
            elif "last_episode_return" in locals():
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
