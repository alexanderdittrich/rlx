"""
Single-file SAC implementation using Flax NNX

Supports continuous action spaces from Gymnasium.
Uses functional programming style with NNX for clarity and maintainability.

References:
- Haarnoja et al. (2018) "Soft Actor-Critic Algorithms and Applications"
- CleanRL SAC implementation
- Stable-Baselines3 hyperparameters
"""

from __future__ import annotations

# Suppress warnings before imports
import os
import warnings

os.environ["JAX_PLATFORMS"] = "cpu"

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")

import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from omegaconf import OmegaConf

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
class SACConfig:
    # Environment
    env_id: str = "HalfCheetah-v5"
    num_envs: int = 1  # SAC typically uses single env

    # Training
    total_timesteps: int = 1_000_000
    learning_starts: int = 5000  # steps before training starts
    batch_size: int = 256  # minibatch size for training

    # SAC hyperparameters
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # soft update coefficient
    target_update_interval: int = 1  # update target networks every N steps
    auto_tune_alpha: bool = True  # automatically tune entropy coefficient
    alpha: float = 0.2  # initial entropy coefficient (if not auto-tuning)
    target_entropy_scale: float = 1.0  # scale for target entropy (-dim(A) * scale)

    # Replay buffer
    buffer_size: int = 1_000_000

    # Network architecture
    actor_hidden_sizes: list[int] | None = None  # Actor/policy network hidden layers
    critic_hidden_sizes: list[int] | None = None  # Critic/Q network hidden layers
    actor_activation: str = "relu"  # Actor/policy network activation
    critic_activation: str = "relu"  # Critic/Q network activation

    # Normalization
    norm_obs: bool = False  # Normalize observations

    # Logging
    log_frequency: int = 1000  # log every N steps
    use_wandb: bool = False  # Enable wandb logging
    wandb_project: str = "sac-nnx"  # Wandb project name
    seed: int = 42

    # Checkpointing
    save_model: bool = True  # Save model checkpoints
    checkpoint_dir: str = "checkpoints"  # Directory to save checkpoints
    checkpoint_frequency: int = 50000  # Save checkpoint every N steps
    keep_checkpoints: int = 3  # Number of checkpoints to keep
    resume_from: str | None = None  # Path to checkpoint directory to resume from

    def __post_init__(self):
        if self.actor_hidden_sizes is None:
            self.actor_hidden_sizes = [256, 256]
        if self.critic_hidden_sizes is None:
            self.critic_hidden_sizes = [256, 256]


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
    """Build an MLP with orthogonal initialization.

    Args:
        in_features: Input dimension
        hidden_sizes: List of hidden layer sizes
        output_size: Output dimension
        output_scale: Orthogonal initialization scale for output layer
        activation_fn: Activation function (nnx.relu, nnx.tanh, etc.)
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


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SquashedGaussianActor(nnx.Module):
    """Squashed Gaussian policy network for continuous action spaces.

    Outputs mean and log_std, samples actions from Gaussian, then applies tanh squashing.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        hidden_sizes: list[int],
        activation_fn: callable,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        in_features = int(np.prod(obs_shape))

        # Shared network
        self.net = _build_mlp(
            in_features,
            hidden_sizes,
            hidden_sizes[-1],  # output is last hidden size
            output_scale=1.0,
            activation_fn=activation_fn,
            rngs=rngs,
        )

        # Mean and log_std heads
        self.mean_layer = nnx.Linear(
            hidden_sizes[-1],
            action_dim,
            kernel_init=nnx.initializers.orthogonal(scale=0.01),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

        self.log_std_layer = nnx.Linear(
            hidden_sizes[-1],
            action_dim,
            kernel_init=nnx.initializers.orthogonal(scale=0.01),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass returning mean and log_std.

        Args:
            x: Observation tensor [batch, ...]

        Returns:
            (mean, log_std): Both [batch, action_dim]
        """
        # Flatten observation if needed
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        features = self.net(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def get_action(
        self, x: jax.Array, key: jax.Array, deterministic: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        """Sample action from the policy.

        Args:
            x: Observation tensor [batch, ...]
            key: Random key for sampling
            deterministic: If True, return mean action (no noise)

        Returns:
            action: Squashed action in [-1, 1] [batch, action_dim]
            log_prob: Log probability of the action [batch]
        """
        mean, log_std = self(x)

        if deterministic:
            # Use mean action for evaluation
            action = jnp.tanh(mean)
            # Log prob is not meaningful for deterministic actions
            log_prob = jnp.zeros(mean.shape[0])
        else:
            # Sample from Gaussian
            std = jnp.exp(log_std)
            normal_sample = mean + std * jax.random.normal(key, mean.shape)

            # Apply tanh squashing
            action = jnp.tanh(normal_sample)

            # Compute log probability with tanh correction
            # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
            log_prob = -0.5 * (
                jnp.sum(jnp.square((normal_sample - mean) / (std + 1e-6)), axis=-1)
                + jnp.sum(2 * jnp.log(std + 1e-6), axis=-1)
                + self.action_dim * jnp.log(2 * jnp.pi)
            )
            # Tanh correction
            log_prob -= jnp.sum(jnp.log(1 - jnp.square(action) + 1e-6), axis=-1)

        return action, log_prob


class QNetwork(nnx.Module):
    """Q-network (critic) for SAC.

    Takes observation and action as input, outputs Q-value.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        hidden_sizes: list[int],
        activation_fn: callable,
        rngs: nnx.Rngs,
    ):
        in_features = int(np.prod(obs_shape)) + action_dim

        self.net = _build_mlp(
            in_features,
            hidden_sizes,
            1,  # single Q-value output
            output_scale=1.0,
            activation_fn=activation_fn,
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Forward pass returning Q-value.

        Args:
            obs: Observation tensor [batch, ...]
            action: Action tensor [batch, action_dim]

        Returns:
            q_value: [batch]
        """
        # Flatten observation if needed
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)

        x = jnp.concatenate([obs, action], axis=-1)
        return self.net(x).squeeze(-1)


class SACNetworks(nnx.Module):
    """Container for all SAC networks: actor, 2 critics, 2 target critics."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        actor_hidden_sizes: list[int],
        critic_hidden_sizes: list[int],
        actor_activation_fn: callable,
        critic_activation_fn: callable,
        rngs: nnx.Rngs,
    ):
        # Actor
        self.actor = SquashedGaussianActor(
            obs_shape, action_dim, actor_hidden_sizes, actor_activation_fn, rngs
        )

        # Twin Q-networks (critics)
        self.qf1 = QNetwork(
            obs_shape, action_dim, critic_hidden_sizes, critic_activation_fn, rngs
        )
        self.qf2 = QNetwork(
            obs_shape, action_dim, critic_hidden_sizes, critic_activation_fn, rngs
        )

        # Target Q-networks
        self.qf1_target = QNetwork(
            obs_shape, action_dim, critic_hidden_sizes, critic_activation_fn, rngs
        )
        self.qf2_target = QNetwork(
            obs_shape, action_dim, critic_hidden_sizes, critic_activation_fn, rngs
        )

        # Copy weights to target networks
        nnx.update(self.qf1_target, nnx.state(self.qf1))
        nnx.update(self.qf2_target, nnx.state(self.qf2))


class CriticPair(nnx.Module):
    """Container for the two trainable Q-networks (for optimizer)."""

    def __init__(self, qf1: QNetwork, qf2: QNetwork):
        self.qf1 = qf1
        self.qf2 = qf2


class Alpha(nnx.Module):
    """Temperature parameter (alpha) for SAC.

    Simple module wrapping log_alpha to work with nnx.Optimizer.
    Follows Brax SAC structure.
    """

    def __init__(self):
        # Initialize log_alpha to 0.0 (alpha = 1.0)
        self.log_alpha = nnx.Param(jnp.array(0.0, dtype=jnp.float32))


# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    """Simple replay buffer for SAC."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

        # Pre-allocate arrays
        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size,) + action_shape, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = obs
        self.next_observations[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> dict[str, jax.Array]:
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": jnp.array(self.observations[idxs]),
            "next_observations": jnp.array(self.next_observations[idxs]),
            "actions": jnp.array(self.actions[idxs]),
            "rewards": jnp.array(self.rewards[idxs]),
            "dones": jnp.array(self.dones[idxs]),
        }


# ---------------------------
# SAC Update Functions
# ---------------------------
@nnx.jit
def update_critic(
    networks: SACNetworks,
    critic_pair: CriticPair,
    critic_optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    alpha: jax.Array,
    gamma: float,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Update critic networks using Bellman backup with target networks."""

    def critic_loss_fn(critics: CriticPair):
        # Sample actions from current policy for next states
        next_actions, next_log_probs = networks.actor.get_action(
            batch["next_observations"], key, deterministic=False
        )

        # Compute target Q-values using target networks
        qf1_next_target = networks.qf1_target(batch["next_observations"], next_actions)
        qf2_next_target = networks.qf2_target(batch["next_observations"], next_actions)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)

        # Bellman backup with entropy term
        next_q_value = min_qf_next_target - alpha * next_log_probs
        target_q_value = batch["rewards"] + (1 - batch["dones"]) * gamma * next_q_value

        # MSE loss for both critics
        qf1_values = critics.qf1(batch["observations"], batch["actions"])
        qf2_values = critics.qf2(batch["observations"], batch["actions"])

        qf1_loss = jnp.mean((qf1_values - target_q_value) ** 2)
        qf2_loss = jnp.mean((qf2_values - target_q_value) ** 2)

        total_loss = qf1_loss + qf2_loss

        return total_loss, {
            "qf1_values": jnp.mean(qf1_values),
            "qf2_values": jnp.mean(qf2_values),
            "qf1_loss": qf1_loss,
            "qf2_loss": qf2_loss,
        }

    # Compute gradients and update
    (loss, info), grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(critic_pair)
    critic_optimizer.update(model=critic_pair, grads=grads)

    return info


@nnx.jit
def update_actor(
    networks: SACNetworks,
    actor_optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    alpha: jax.Array,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Update actor network to maximize Q-value - entropy."""

    def actor_loss_fn(actor: SquashedGaussianActor):
        # Sample actions from current policy
        actions, log_probs = actor.get_action(
            batch["observations"], key, deterministic=False
        )

        # Compute Q-values for sampled actions
        qf1_values = networks.qf1(batch["observations"], actions)
        qf2_values = networks.qf2(batch["observations"], actions)
        min_qf_values = jnp.minimum(qf1_values, qf2_values)

        # Actor loss: maximize Q - α * entropy
        actor_loss = jnp.mean(alpha * log_probs - min_qf_values)

        return actor_loss, {
            "actor_loss": actor_loss,
            "log_probs": jnp.mean(log_probs),
        }

    # Compute gradients and update
    (loss, info), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(
        networks.actor
    )
    actor_optimizer.update(networks.actor, grads)

    return info


@nnx.jit
def update_alpha(
    alpha_module: Alpha,
    alpha_optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
    target_entropy: float,
    networks: SACNetworks,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Update temperature parameter alpha.

    Following Brax SAC implementation.
    """

    def alpha_loss_fn(alpha_mod: Alpha):
        # Compute current alpha
        alpha = jnp.exp(alpha_mod.log_alpha.value)

        # Sample actions to get current policy entropy
        _, log_probs = networks.actor.get_action(
            batch["observations"], key, deterministic=False
        )

        # Alpha loss: Eq 18 from https://arxiv.org/pdf/1812.05905.pdf
        # alpha * stop_gradient(-log_prob - target_entropy)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_probs - target_entropy)
        alpha_loss = jnp.mean(alpha_loss)

        return alpha_loss, {"alpha_loss": alpha_loss, "alpha": alpha}

    # Compute gradients and update
    (loss, info), grads = nnx.value_and_grad(alpha_loss_fn, has_aux=True)(alpha_module)
    alpha_optimizer.update(alpha_module, grads)

    # Return updated alpha value
    alpha = jnp.exp(alpha_module.log_alpha.value)
    return alpha, info


@jax.jit
def soft_update_targets(
    qf1: QNetwork,
    qf2: QNetwork,
    qf1_target: QNetwork,
    qf2_target: QNetwork,
    tau: float,
):
    """Soft update target networks: θ_target = τ*θ + (1-τ)*θ_target"""
    # Get current states
    qf1_state = nnx.state(qf1)
    qf2_state = nnx.state(qf2)
    qf1_target_state = nnx.state(qf1_target)
    qf2_target_state = nnx.state(qf2_target)

    # Perform soft update
    new_qf1_target_state = jax.tree.map(
        lambda x, y: tau * x + (1 - tau) * y, qf1_state, qf1_target_state
    )
    new_qf2_target_state = jax.tree.map(
        lambda x, y: tau * x + (1 - tau) * y, qf2_state, qf2_target_state
    )

    # Update target networks
    nnx.update(qf1_target, new_qf1_target_state)
    nnx.update(qf2_target, new_qf2_target_state)


# ---------------------------
# Checkpoint utilities
# ---------------------------
def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    networks: SACNetworks,
    alpha_module: Alpha | None,
    global_step: int,
    key: jax.Array,
    metrics: dict,
) -> None:
    """Save a checkpoint with the current model state and training progress."""
    checkpoint_state = {
        "networks_state": nnx.state(networks),
        "alpha_state": nnx.state(alpha_module) if alpha_module is not None else None,
        "global_step": global_step,
        "rng_key": key,
    }

    checkpoint_manager.save(
        step=global_step,
        args=ocp.args.StandardSave(checkpoint_state),
        metrics=metrics,
    )
    checkpoint_manager.wait_until_finished()


def load_checkpoint(
    checkpoint_path: str,
    networks: SACNetworks,
    alpha_module: Alpha | None,
) -> tuple[int, jax.Array]:
    """Load a checkpoint and restore model state.

    Returns:
        global_step: Global step count
        rng_key: RNG key state
    """
    checkpoint_manager = ocp.CheckpointManager(
        directory=str(Path(checkpoint_path).resolve()),
        options=ocp.CheckpointManagerOptions(create=False),
    )

    # Get the latest checkpoint
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError(f"No checkpoint found in {checkpoint_path}")

    # Create target structure for restoration
    target_checkpoint = {
        "networks_state": nnx.state(networks),
        "alpha_state": nnx.state(alpha_module) if alpha_module is not None else None,
        "global_step": 0,
        "rng_key": jax.random.key(0),
    }

    checkpoint_state = checkpoint_manager.restore(
        latest_step, args=ocp.args.StandardRestore(target_checkpoint)
    )

    # Restore network and alpha states
    nnx.update(networks, checkpoint_state["networks_state"])
    if alpha_module is not None and checkpoint_state["alpha_state"] is not None:
        nnx.update(alpha_module, checkpoint_state["alpha_state"])

    print(f"Loaded checkpoint from step {latest_step}")
    print(f"  Global step: {checkpoint_state['global_step']}")

    return checkpoint_state["global_step"], checkpoint_state["rng_key"]


# ---------------------------
# Main training loop
# ---------------------------
def train(cfg: SACConfig):
    """Main SAC training loop."""
    # Set random seeds
    np.random.seed(cfg.seed)
    key = jax.random.key(cfg.seed)

    # Create environment
    envs = gym.make_vec(
        cfg.env_id,
        num_envs=cfg.num_envs,
        vectorization_mode="sync",
    )

    # Seed the environment
    envs.reset(seed=cfg.seed)

    # Apply wrappers
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    # SAC only works with continuous action spaces
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "SAC only supports continuous action spaces"
    )

    # Normalize action space to [-1, 1]
    envs = gym.wrappers.vector.RescaleAction(envs, -1, 1)

    if cfg.norm_obs:
        envs = gym.wrappers.vector.NormalizeObservation(envs)
        envs = gym.wrappers.vector.TransformObservation(
            envs, lambda obs: np.clip(obs, -10, 10)
        )

    # Initialize networks
    obs_shape = envs.single_observation_space.shape
    action_dim = int(np.prod(envs.single_action_space.shape))

    # Get activation functions
    actor_activation_fn = get_activation_fn(cfg.actor_activation)
    critic_activation_fn = get_activation_fn(cfg.critic_activation)

    key, networks_key = jax.random.split(key)
    networks = SACNetworks(
        obs_shape=obs_shape,
        action_dim=action_dim,
        actor_hidden_sizes=cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        actor_activation_fn=actor_activation_fn,
        critic_activation_fn=critic_activation_fn,
        rngs=nnx.Rngs(networks_key),
    )

    # Initialize optimizers
    actor_optimizer = nnx.Optimizer(
        networks.actor, optax.adam(learning_rate=cfg.actor_lr), wrt=nnx.Param
    )
    # Critic optimizer optimizes both Q-networks via CriticPair container
    critic_pair = CriticPair(networks.qf1, networks.qf2)
    critic_optimizer = nnx.Optimizer(
        critic_pair,
        optax.adam(learning_rate=cfg.critic_lr),
        wrt=nnx.Param,
    )

    # Initialize alpha (temperature parameter)
    # Following Brax SAC structure
    if cfg.auto_tune_alpha:
        # Target entropy: -dim(A) * scale (Brax uses -0.5 * action_dim by default)
        target_entropy = -action_dim * cfg.target_entropy_scale
        alpha_module = Alpha()
        alpha_optimizer = nnx.Optimizer(
            alpha_module, optax.adam(learning_rate=cfg.alpha_lr), wrt=nnx.Param
        )
        alpha = jnp.exp(alpha_module.log_alpha.value)
    else:
        target_entropy = None
        alpha_module = None
        alpha_optimizer = None
        alpha = jnp.array(cfg.alpha)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        buffer_size=cfg.buffer_size,
        obs_shape=obs_shape,
        action_shape=envs.single_action_space.shape,
    )

    # Create run ID with timestamp
    time_stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    env_name_clean = re.sub(r"-v\d+$", "", cfg.env_id.lower())
    run_id = f"{time_stamp}-{env_name_clean}-sac-{cfg.seed}"

    # Setup checkpointing
    checkpoint_manager = None
    if cfg.save_model:
        checkpoint_path = Path(cfg.checkpoint_dir) / run_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = ocp.CheckpointManager(
            directory=str(checkpoint_path.resolve()),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=cfg.keep_checkpoints,
                create=True,
            ),
        )
        print(f"Checkpoint directory: {checkpoint_path}")

    # Initialize wandb
    if cfg.use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb_project, name=run_id, config=OmegaConf.to_container(cfg)
        )

    # Training loop
    obs, _ = envs.reset(seed=cfg.seed)
    global_step = 0

    # Resume from checkpoint if specified
    if cfg.resume_from is not None:
        global_step, key = load_checkpoint(cfg.resume_from, networks, alpha_module)

    # Episode tracking
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)

    start_time = time.time()

    while global_step < cfg.total_timesteps:
        # Collect experience
        if global_step < cfg.learning_starts:
            # Random actions for initial exploration
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(cfg.num_envs)]
            )
        else:
            # Sample actions from policy
            key, action_key = jax.random.split(key)
            actions_jax, _ = networks.actor.get_action(
                jnp.array(obs), action_key, deterministic=False
            )
            actions = np.array(actions_jax)

        # Step environment
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        # Store transitions in replay buffer
        for i in range(cfg.num_envs):
            replay_buffer.add(obs[i], next_obs[i], actions[i], rewards[i], dones[i])

        obs = next_obs
        global_step += cfg.num_envs

        # Log episode statistics
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    episode_returns.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])

        # Train agent
        if global_step >= cfg.learning_starts:
            # Sample batch from replay buffer
            batch = replay_buffer.sample(cfg.batch_size)

            # Update critics
            key, critic_key = jax.random.split(key)
            critic_info = update_critic(
                networks,
                critic_pair,
                critic_optimizer,
                batch,
                alpha,
                cfg.gamma,
                critic_key,
            )

            # Update actor
            key, actor_key = jax.random.split(key)
            actor_info = update_actor(
                networks, actor_optimizer, batch, alpha, actor_key
            )

            # Update alpha if auto-tuning
            if cfg.auto_tune_alpha:
                key, alpha_key = jax.random.split(key)
                alpha, alpha_info = update_alpha(
                    alpha_module,
                    alpha_optimizer,
                    batch,
                    target_entropy,
                    networks,
                    alpha_key,
                )
            else:
                alpha_info = {"alpha": alpha}

            # Soft update target networks
            if global_step % cfg.target_update_interval == 0:
                soft_update_targets(
                    networks.qf1,
                    networks.qf2,
                    networks.qf1_target,
                    networks.qf2_target,
                    cfg.tau,
                )

            # Logging
            if global_step % cfg.log_frequency == 0:
                sps = int(global_step / (time.time() - start_time))

                metrics = {
                    "global_step": global_step,
                    "sps": sps,
                    **critic_info,
                    **actor_info,
                    **alpha_info,
                }

                if len(episode_returns) > 0:
                    metrics["episode/return_mean"] = np.mean(episode_returns)
                    metrics["episode/return_std"] = np.std(episode_returns)
                    metrics["episode/length_mean"] = np.mean(episode_lengths)

                if cfg.use_wandb:
                    wandb.log(metrics, step=global_step)

                # Console output
                print(f"Step {global_step}/{cfg.total_timesteps} | SPS: {sps}")
                if len(episode_returns) > 0:
                    print(
                        f"  Episode Return: {metrics['episode/return_mean']:.2f} ± {metrics['episode/return_std']:.2f}"
                    )
                print(
                    f"  Actor Loss: {actor_info['actor_loss']:.4f} | "
                    f"Q1 Loss: {critic_info['qf1_loss']:.4f} | "
                    f"Alpha: {alpha_info['alpha']:.4f}"
                )

        # Save checkpoint
        if cfg.save_model and global_step % cfg.checkpoint_frequency == 0:
            if checkpoint_manager is not None and global_step > 0:
                checkpoint_metrics = {}
                if len(episode_returns) > 0:
                    checkpoint_metrics["return_mean"] = float(np.mean(episode_returns))

                save_checkpoint(
                    checkpoint_manager,
                    networks,
                    alpha_module,
                    global_step,
                    key,
                    checkpoint_metrics,
                )
                print(f"  Saved checkpoint at step {global_step}")

    # Save final checkpoint
    if cfg.save_model and checkpoint_manager is not None:
        checkpoint_metrics = {}
        if len(episode_returns) > 0:
            checkpoint_metrics["return_mean"] = float(np.mean(episode_returns))

        save_checkpoint(
            checkpoint_manager,
            networks,
            alpha_module,
            global_step,
            key,
            checkpoint_metrics,
        )
        print(f"Saved final checkpoint at step {global_step}")

    envs.close()

    if cfg.use_wandb:
        wandb.finish()

    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
