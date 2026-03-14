"""
Single-file PPO implementation using Flax NNX

Supports both discrete and continuous action spaces from Gymnasium.
Uses functional programming style with NNX for clarity and maintainability.

References:
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- CleanRL PPO implementation
- Stable-Baselines3 hyperparameters
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

import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx

# Set GPU parameters
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Timestamp format for run IDs
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


# ---------------------------
# Configuration
# ---------------------------
@dataclass
class PPOConfig:
    # Environment
    env_id: str = "CartPole-v1"
    num_envs: int = 4

    # Training
    total_timesteps: int = 500_000
    num_steps: int = 2048  # rollout length per env
    batch_size: int = 64  # minibatch size for training
    update_epochs: int = 10  # n_epochs in SB3

    # PPO hyperparameters
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True  # Anneal learning rate
    lr_schedule_type: str = "linear"  # linear, exponential, constant
    decay_rate: float = 0.99  # for exponential decay
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01  # 0.01 for more exploration
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
    norm_obs: bool = False  # Normalize observations
    norm_reward: bool = False  # Normalize rewards
    clip_obs: float = 10.0  # Clip normalized observations
    clip_reward: float = 10.0  # Clip normalized rewards

    # Logging
    log_frequency: int = 10  # log every N updates
    use_wandb: bool = False  # Enable wandb logging
    wandb_project: str = "ppo-nnx"  # Wandb project name
    seed: int = 42

    # Checkpointing
    save_model: bool = True  # Save model checkpoints
    checkpoint_dir: str = "checkpoints"  # Directory to save checkpoints
    checkpoint_frequency: int = 100  # Save checkpoint every N updates
    keep_checkpoints: int = 3  # Number of checkpoints to keep
    resume_from: str | None = None  # Path to checkpoint directory to resume from

    def __post_init__(self):
        # Set default hidden sizes if not provided
        if self.actor_hidden_sizes is None:
            self.actor_hidden_sizes = [64, 64]
        if self.critic_hidden_sizes is None:
            self.critic_hidden_sizes = [64, 64]

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
                kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
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

    activation_lower = activation.lower()
    if activation_lower not in activations:
        raise ValueError(
            f"Unknown activation function: {activation}. "
            f"Supported: {list(activations.keys())}"
        )

    return activations[activation_lower]


class ActorCritic(nnx.Module):
    """Actor-critic network with separate policy and value networks.

    Supports both discrete and continuous action spaces, matching SB3 architecture.
    Allows different architectures (hidden sizes and activations) for actor and critic.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_space: gym.Space,
        actor_hidden_sizes: list[int],
        critic_hidden_sizes: list[int],
        actor_activation_fn: callable,
        critic_activation_fn: callable,
        rngs: nnx.Rngs,
    ):
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        in_features = int(np.prod(obs_shape))

        # Determine output dimensions
        if self.is_discrete:
            action_dim = action_space.n
        else:
            action_dim = int(np.prod(action_space.shape))
            # Learnable log standard deviation for continuous actions
            self.action_logstd = nnx.Param(jnp.zeros(action_dim))

        # Build policy network (actor for continuous, policy for discrete)
        self.policy_net = _build_mlp(
            in_features,
            actor_hidden_sizes,
            action_dim,
            output_scale=0.01,
            activation_fn=actor_activation_fn,
            rngs=rngs,
        )

        # Build value network (separate from policy, matching SB3)
        self.value_net = _build_mlp(
            in_features,
            critic_hidden_sizes,
            1,
            output_scale=1.0,
            activation_fn=critic_activation_fn,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[distrax.Distribution, jax.Array]:
        """Forward pass returning action distribution and value estimate.

        Args:
            x: Observation tensor [batch, ...]

        Returns:
            (action_distribution, value_estimate)
        """
        # Flatten observation if needed
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        # Get policy output (logits for discrete, mean for continuous)
        policy_output = self.policy_net(x)
        value = self.value_net(x).squeeze(-1)

        if self.is_discrete:
            # Discrete: Categorical distribution over logits
            dist = distrax.Categorical(logits=policy_output)
        else:
            # Continuous: Diagonal Gaussian with learnable std
            log_std = jnp.broadcast_to(self.action_logstd.value, policy_output.shape)
            std = jnp.exp(log_std)
            dist = distrax.MultivariateNormalDiag(loc=policy_output, scale_diag=std)

        return dist, value

    def get_value(self, x: jax.Array) -> jax.Array:
        """Get value estimate only (used for bootstrapping).

        Args:
            x: Observation tensor [batch, ...]

        Returns:
            value_estimate: [batch]
        """
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        return self.value_net(x).squeeze(-1)

    def get_action_and_value(
        self,
        x: jax.Array,
        action: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Get action (sampled or provided) with its log probability, entropy, and value.

        Args:
            x: Observation tensor [batch, ...]
            action: Optional action to evaluate. If None, samples from policy.
            key: Random key for sampling (required if action is None)

        Returns:
            action: Sampled or provided action [batch, ...]
            log_prob: Log probability of the action [batch]
            entropy: Entropy of the action distribution [batch]
            value: Value estimate [batch]
        """
        dist, value = self(x)

        if action is None:
            action = dist.sample(seed=key)

        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, action_log_prob, entropy, value


# ---------------------------
# Jitted inference functions
# ---------------------------
@jax.jit
def predict_action_and_value(
    model: ActorCritic,
    obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Jitted function for sampling actions during rollout collection.

    Args:
        model: The actor-critic model
        obs: Observations [batch, ...]
        key: Random key for action sampling

    Returns:
        action: Sampled actions [batch, ...]
        log_prob: Log probabilities [batch]
        value: Value estimates [batch]
    """
    dist, value = model(obs)
    action = dist.sample(seed=key)
    log_prob = dist.log_prob(action)
    return action, log_prob, value


@jax.jit
def predict_value(model: ActorCritic, obs: jax.Array) -> jax.Array:
    """Jitted function for computing values (used for bootstrapping).

    Args:
        model: The actor-critic model
        obs: Observations [batch, ...]

    Returns:
        value: Value estimates [batch]
    """
    return model.get_value(obs)


# ---------------------------
# GAE and advantages
# ---------------------------
@jax.jit
def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    next_done: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute Generalized Advantage Estimation using jax.lax.scan for efficiency.

    Args:
        rewards: [num_steps, num_envs]
        values: [num_steps, num_envs]
        dones: [num_steps, num_envs] - terminal state at start of each step
        next_value: [num_envs] - value estimate for state after last step
        next_done: [num_envs] - terminal state after last step
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: [num_steps, num_envs]
        returns: [num_steps, num_envs]
    """
    # Prepare data for backward scan
    # Append next_value and next_done to the end
    values_with_next = jnp.concatenate([values, next_value[None, :]], axis=0)
    dones_with_next = jnp.concatenate([dones, next_done[None, :]], axis=0)

    def scan_fn(gae, t):
        """Scan function for GAE computation (processes timesteps backwards)."""
        reward = rewards[t]
        value = values[t]
        next_value_t = values_with_next[t + 1]
        next_nonterminal = 1.0 - dones_with_next[t + 1]

        # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        delta = reward + gamma * next_value_t * next_nonterminal - value

        # GAE: A_t = δ_t + γλ * (1 - done_{t+1}) * A_{t+1}
        gae = delta + gamma * gae_lambda * next_nonterminal * gae

        return gae, gae

    # Scan backwards through time to compute advantages
    # Start with lastgaelam = 0 and process from last step to first
    num_steps = rewards.shape[0]
    _, advantages = jax.lax.scan(
        scan_fn,
        init=jnp.zeros_like(next_value),  # Initial GAE (for step after last)
        xs=jnp.arange(num_steps - 1, -1, -1),  # Timesteps in reverse order
    )

    # Reverse advantages back to forward order
    advantages = advantages[::-1]

    # Returns are advantages + values
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

    # Clipped surrogate objective
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss - use jax.lax.cond for JIT compatibility
    def compute_clipped_v_loss():
        v_loss_unclipped = (new_values - returns) ** 2
        v_clipped = old_values + jnp.clip(
            new_values - old_values, -clip_coef, clip_coef
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        return 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

    def compute_unclipped_v_loss():
        return 0.5 * ((new_values - returns) ** 2).mean()

    v_loss = jax.lax.cond(clip_vloss, compute_clipped_v_loss, compute_unclipped_v_loss)

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
    """Single training step (advantages should be pre-normalized if needed)."""
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
    """
    Save a checkpoint with the current model state and training progress.

    Args:
        checkpoint_manager: Orbax checkpoint manager
        model: Actor-critic model
        global_step: Current global step count
        num_updates: Current number of updates
        key: Current RNG key state
        metrics: Metrics to save with checkpoint
    """
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
    optimizer: nnx.Optimizer,
) -> tuple[int, int, jax.Array]:
    """
    Load a checkpoint and restore model and optimizer state.

    Returns:
        global_step: Global step count
        num_updates: Number of updates completed
        rng_key: RNG key state
    """
    checkpoint_manager = ocp.CheckpointManager(
        directory=str(Path(checkpoint_path).resolve()),
        options=ocp.CheckpointManagerOptions(create=False),
    )

    # Get the latest checkpoint
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError(f"No checkpoints found in {checkpoint_path}")

    # Load checkpoint using new API - provide target structure
    # First get a template of the current model state structure
    template_state = nnx.state(model)

    # Create the target structure for restoration
    target_checkpoint = {
        "model_state": template_state,
        "global_step": 0,
        "num_updates": 0,
        "rng_key": jax.random.key(0),
    }

    checkpoint_state = checkpoint_manager.restore(
        latest_step, args=ocp.args.StandardRestore(target_checkpoint)
    )

    # Restore model state (optimizer will be recreated)
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
def train(cfg: PPOConfig):
    """Main PPO training loop."""
    # Set random seeds
    np.random.seed(cfg.seed)
    key = jax.random.key(cfg.seed)

    # Create vectorized environment using gym.make_vec
    envs = gym.make_vec(
        cfg.env_id,
        num_envs=cfg.num_envs,
        vectorization_mode="sync",
    )

    # Seed the environments
    envs.reset(seed=cfg.seed)

    # Apply vectorized normalization wrappers if requested
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    if isinstance(envs.single_action_space, gym.spaces.Box):
        envs = gym.wrappers.vector.ClipAction(envs)

    if cfg.norm_obs:
        envs = gym.wrappers.vector.NormalizeObservation(envs)
        envs = gym.wrappers.vector.TransformObservation(
            envs, lambda obs: np.clip(obs, -cfg.clip_obs, cfg.clip_obs)
        )
    if cfg.norm_reward:
        envs = gym.wrappers.vector.NormalizeReward(envs, gamma=cfg.gamma)
        envs = gym.wrappers.vector.ClipReward(
            envs, min_reward=-cfg.clip_reward, max_reward=cfg.clip_reward
        )

    # Initialize model
    obs_shape = envs.single_observation_space.shape
    action_space = envs.single_action_space

    # Get activation functions for actor and critic
    actor_activation_fn = get_activation_fn(cfg.actor_activation)
    critic_activation_fn = get_activation_fn(cfg.critic_activation)

    key, model_key = jax.random.split(key)
    model = ActorCritic(
        obs_shape=obs_shape,
        action_space=action_space,
        actor_hidden_sizes=cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        actor_activation_fn=actor_activation_fn,
        critic_activation_fn=critic_activation_fn,
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
                alpha=0.0,  # Final learning rate as fraction of initial
            )
        elif cfg.lr_schedule_type == "constant":
            lr_schedule = optax.constant_schedule(cfg.learning_rate)
        else:
            raise ValueError(f"Unknown lr_schedule_type: {cfg.lr_schedule_type}")
    else:
        lr_schedule = optax.constant_schedule(cfg.learning_rate)

    # Initialize optimizer with schedule and gradient clipping
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
    # Remove version suffix from environment name (e.g., -v1, -v5)
    env_name_clean = re.sub(r"-v\d+$", "", cfg.env_id.lower())
    run_id = f"{time_stamp}-{env_name_clean}-ppo-{cfg.seed}"

    # Setup checkpointing
    checkpoint_manager = None
    if cfg.save_model:
        checkpoint_dir = Path(cfg.checkpoint_dir).resolve() / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoint manager using new API
        checkpoint_manager = ocp.CheckpointManager(
            directory=str(checkpoint_dir),  # Ensure absolute path as string
            options=ocp.CheckpointManagerOptions(
                max_to_keep=cfg.keep_checkpoints,
                create=True,
            ),
        )

    # Initialize wandb
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            config=vars(cfg),  # Convert dataclass to dict
            name=run_id,
        )

    # Training loop
    obs, _ = envs.reset(seed=cfg.seed)
    global_step = 0
    num_updates = 0

    # Resume from checkpoint if specified
    if cfg.resume_from is not None:
        global_step, num_updates, key = load_checkpoint(
            cfg.resume_from, model, optimizer
        )

    # Storage for rollouts
    # Use NumPy for data that interacts with Gym (CPU-based)
    next_done = np.zeros(cfg.num_envs, dtype=np.float32)

    rollout_obs = np.zeros((cfg.num_steps, cfg.num_envs) + obs_shape, dtype=np.float32)
    rollout_actions = np.zeros(
        (cfg.num_steps, cfg.num_envs) + action_space.shape, dtype=np.float32
    )
    rollout_rewards = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.float32)
    rollout_dones = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.float32)

    # Use JAX arrays for data that never needs to be NumPy (stays on GPU)
    rollout_logprobs = jnp.zeros((cfg.num_steps, cfg.num_envs), dtype=jnp.float32)
    rollout_values = jnp.zeros((cfg.num_steps, cfg.num_envs), dtype=jnp.float32)

    # Episode tracking
    episode_returns = []
    episode_lengths = []

    start_time = time.time()

    for iteration in range(cfg.num_iterations):
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            # Store in NumPy arrays (mutable, fast in-place updates)
            rollout_obs[step] = obs
            rollout_dones[step] = next_done

            # Sample action using jitted function (JAX handles NumPy->GPU transfer automatically)
            key, action_key = jax.random.split(key)
            action, logprob, value = predict_action_and_value(model, obs, action_key)

            # Store action as NumPy (needed for env.step)
            action_np = np.array(action)
            rollout_actions[step] = action_np

            # Store logprobs and values in JAX arrays (stay on GPU, no round-trip)
            rollout_logprobs = rollout_logprobs.at[step].set(logprob)
            rollout_values = rollout_values.at[step].set(value)

            # Step environment with already-converted action (no additional sync)
            obs, reward, terminated, truncated, info = envs.step(action_np)
            next_done = terminated | truncated

            # Store rewards in NumPy array
            rollout_rewards[step] = reward

            # Log episode stats - gym.make_vec uses "episode" and "_episode" keys
            if "_episode" in info and info["_episode"].any():
                # Vector wrapper format
                episode_mask = info["_episode"]
                if "episode" in info:
                    episode_data = info["episode"]
                    for idx in np.where(episode_mask)[0]:
                        episode_returns.append(float(episode_data["r"][idx]))
                        episode_lengths.append(int(episode_data["l"][idx]))
            elif "final_info" in info:
                # Old format fallback
                for env_info in info["final_info"]:
                    if env_info is not None and "episode" in env_info:
                        episode_returns.append(env_info["episode"]["r"])
                        episode_lengths.append(env_info["episode"]["l"])

        # Transfer only NumPy arrays to GPU (logprobs and values already on GPU!)
        rollout_actions_jax = jnp.array(rollout_actions)
        rollout_obs_jax = jnp.array(rollout_obs)
        rollout_rewards_jax = jnp.array(rollout_rewards)
        rollout_dones_jax = jnp.array(rollout_dones)
        next_done_jax = jnp.array(next_done)

        # Bootstrap value for GAE (jitted function handles transfer)
        next_value = predict_value(model, obs)

        # Compute advantages and returns (logprobs and values already JAX arrays on GPU)
        advantages, returns = compute_gae(
            rollout_rewards_jax,
            rollout_values,
            rollout_dones_jax,
            next_value,
            next_done_jax,
            cfg.gamma,
            cfg.gae_lambda,
        )

        # Flatten batch
        b_obs = rollout_obs_jax.reshape((-1,) + obs_shape)
        b_actions = rollout_actions_jax.reshape((-1,) + action_space.shape)
        b_logprobs = rollout_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = rollout_values.reshape(-1)

        # Update policy - optimized with JAX arrays and deferred metric sync
        update_info = {}
        for epoch in range(cfg.update_epochs):
            # Random permutation for minibatches
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, cfg.rollout_buffer_size)

            for start in range(0, cfg.rollout_buffer_size, cfg.batch_size):
                mb_inds = perm[start : start + cfg.batch_size]

                # Normalize advantages per minibatch (outside JIT is faster)
                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                info = train_step(
                    model=model,
                    optimizer=optimizer,
                    obs=b_obs[mb_inds],
                    actions=b_actions[mb_inds],
                    old_logprobs=b_logprobs[mb_inds],
                    advantages=mb_advantages,
                    returns=b_returns[mb_inds],
                    old_values=b_values[mb_inds],
                    clip_coef=cfg.clip_coef,
                    vf_coef=cfg.vf_coef,
                    ent_coef=cfg.ent_coef,
                    clip_vloss=cfg.clip_vloss,
                )

                # Accumulate info (keep as JAX arrays, don't sync yet)
                for k, v in info.items():
                    update_info[k] = update_info.get(k, 0.0) + v

            # Check KL divergence for early stopping
            if cfg.target_kl is not None:
                avg_kl = update_info.get("loss/approx_kl", 0) / (
                    (epoch + 1) * cfg.num_minibatches
                )
                if avg_kl > 1.5 * cfg.target_kl:
                    print(f"Early stopping at epoch {epoch} due to KL={avg_kl:.4f}")
                    break

        num_updates += 1

        # Average update info and convert to Python floats (sync happens here once per update)
        num_minibatch_updates = (epoch + 1) * cfg.num_minibatches
        for k in update_info:
            update_info[k] = float(update_info[k] / num_minibatch_updates)

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
                log_dict["episode/length_mean"] = np.mean(episode_lengths)

                # Store for checkpoint saving (in case checkpoint happens without new episodes)
                last_episode_return = log_dict["episode/return_mean"]

                episode_returns = []
                episode_lengths = []

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
            # Use the most recent episode return if available
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

    envs.close()

    if cfg.use_wandb:
        wandb.finish()

    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
