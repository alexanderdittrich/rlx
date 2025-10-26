"""
PPO Implementation for Vision-based Observations (CNN-based)

Supports visual input (image-based observations) using CNN feature extractors.
Follows Stable-Baselines3's NatureCNN architecture for Atari and other vision-based environments.

CNN Architecture (from SB3):
- NatureCNN: 3 conv layers + flatten + linear (for 84x84 Atari)
  - Conv1: 32 filters, 8x8 kernel, stride 4, ReLU
  - Conv2: 64 filters, 4x4 kernel, stride 2, ReLU
  - Conv3: 64 filters, 3x3 kernel, stride 1, ReLU
  - Flatten
  - Linear: 512 units, ReLU

References:
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Mnih et al. (2015) "Human-level control through deep reinforcement learning" (DQN Nature paper)
- Stable-Baselines3 CNN policies
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

# Timestamp format for run IDs
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


# ---------------------------
# Configuration
# ---------------------------
@dataclass
class PPOVisionConfig:
    # Environment
    env_id: str = "ALE/Pong-v5"  # Atari environment
    num_envs: int = 8

    # Training
    total_timesteps: int = 10_000_000
    num_steps: int = 128  # rollout length per env
    batch_size: int = 256  # minibatch size for training
    update_epochs: int = 4  # n_epochs in SB3

    # PPO hyperparameters (SB3 Atari defaults)
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    lr_schedule_type: str = "linear"
    decay_rate: float = 0.99  # for exponential decay
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1  # SB3 uses 0.1 for Atari
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # CNN architecture (NatureCNN from SB3)
    cnn_type: str = "nature"  # "nature" or "impala"
    features_dim: int = 512  # Output dimension of CNN feature extractor

    # Policy/Value head architecture (after CNN features)
    actor_hidden_sizes: list[int] | None = None  # Additional MLP layers after CNN
    critic_hidden_sizes: list[int] | None = None  # Additional MLP layers after CNN
    actor_activation: str = "relu"
    critic_activation: str = "relu"

    # Normalization
    norm_adv: bool = True
    norm_obs: bool = False  # Not typically used for visual observations
    norm_reward: bool = True  # Normalize rewards for Atari
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # Frame stacking
    frame_stack: int = 4  # Stack last N frames as input

    # Logging
    log_frequency: int = 10
    use_wandb: bool = False
    wandb_project: str = "ppo-gymnasium-vision"
    seed: int = 42

    # Checkpointing
    save_model: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100
    keep_checkpoints: int = 3
    resume_from: str | None = None

    def __post_init__(self):
        # Set default hidden sizes if not provided (typically empty for visual)
        if self.actor_hidden_sizes is None:
            self.actor_hidden_sizes = []  # No additional layers after CNN
        if self.critic_hidden_sizes is None:
            self.critic_hidden_sizes = []  # No additional layers after CNN

        # Calculate derived values
        self.rollout_buffer_size = self.num_envs * self.num_steps
        self.num_minibatches = self.rollout_buffer_size // self.batch_size
        self.num_iterations = self.total_timesteps // self.rollout_buffer_size

        # Sanity checks
        assert self.rollout_buffer_size > 1 or not self.norm_adv
        assert self.batch_size > 1 or not self.norm_adv


# ---------------------------
# CNN Feature Extractors
# ---------------------------
class NatureCNN(nnx.Module):
    """NatureCNN architecture from DQN Nature paper and used in SB3.

    Standard architecture for Atari (84x84x4 input):
    - Conv1: 32 filters, 8x8 kernel, stride 4, ReLU
    - Conv2: 64 filters, 4x4 kernel, stride 2, ReLU
    - Conv3: 64 filters, 3x3 kernel, stride 1, ReLU
    - Flatten
    - Linear: features_dim units, ReLU

    Args:
        features_dim: Output feature dimension (typically 512)
        rngs: Random number generators
    """

    def __init__(self, features_dim: int = 512, rngs: nnx.Rngs = None):
        # Conv layers with orthogonal initialization (SB3 standard)
        self.conv1 = nnx.Conv(
            in_features=4,  # Frame stack
            out_features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

        self.conv3 = nnx.Conv(
            in_features=64,
            out_features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

        # Linear layer after flattening
        # For 84x84 input: (84-8)/4+1=20, (20-4)/2+1=9, (9-3)/1+1=7
        # Flattened size: 64 * 7 * 7 = 3136
        self.linear = nnx.Linear(
            in_features=3136,
            out_features=features_dim,
            kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input images [batch, height, width, channels]
               Expected: [batch, 84, 84, 4] for Atari with frame stacking

        Returns:
            features: [batch, features_dim]
        """
        # Normalize pixel values to [0, 1]
        x = x.astype(jnp.float32) / 255.0

        # Conv layers with ReLU
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Linear + ReLU
        x = nnx.relu(self.linear(x))

        return x


class ImpalaCNN(nnx.Module):
    """IMPALA CNN architecture (deeper, better for complex visual tasks).

    Used in IMPALA paper and available in SB3. Better for more complex
    visual environments than NatureCNN.

    Architecture:
    - 3 residual blocks, each with:
      - Conv: channels, 3x3 kernel, stride 1
      - MaxPool: 3x3 kernel, stride 2
      - 2x Residual layers (Conv + ReLU + Conv + ReLU + residual connection)
    - Flatten + Linear + ReLU

    Args:
        features_dim: Output feature dimension
        depths: Number of channels for each block [16, 32, 32]
        rngs: Random number generators
    """

    def __init__(
        self,
        features_dim: int = 512,
        depths: list[int] = None,
        rngs: nnx.Rngs = None,
    ):
        if depths is None:
            depths = [16, 32, 32]

        self.depths = depths
        self.blocks = []

        in_channels = 4  # Frame stack
        for depth in depths:
            block = self._make_block(in_channels, depth, rngs)
            self.blocks.append(block)
            in_channels = depth

        # Output size after 3 max pools (stride 2 each): 84 -> 42 -> 21 -> 10 (rounded down)
        # For 84x84 input: 10x10x32 = 3200
        # This is approximate; actual size depends on pooling details
        self.linear = nnx.Linear(
            in_features=3200,  # Approximate flattened size
            out_features=features_dim,
            kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

    def _make_block(self, in_channels: int, out_channels: int, rngs: nnx.Rngs):
        """Create a residual block."""
        return {
            "conv": nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            ),
            "res1_conv1": nnx.Conv(
                in_features=out_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            ),
            "res1_conv2": nnx.Conv(
                in_features=out_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            ),
            "res2_conv1": nnx.Conv(
                in_features=out_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            ),
            "res2_conv2": nnx.Conv(
                in_features=out_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                kernel_init=nnx.initializers.orthogonal(scale=jnp.sqrt(2)),
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            ),
        }

    def _forward_block(self, x: jax.Array, block: dict) -> jax.Array:
        """Forward pass through a residual block."""
        x = block["conv"](x)
        x = jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1, 3, 3, 1),
            window_strides=(1, 2, 2, 1),
            padding="SAME",
        )

        # Residual block 1
        residual = x
        x = nnx.relu(x)
        x = block["res1_conv1"](x)
        x = nnx.relu(x)
        x = block["res1_conv2"](x)
        x = x + residual

        # Residual block 2
        residual = x
        x = nnx.relu(x)
        x = block["res2_conv1"](x)
        x = nnx.relu(x)
        x = block["res2_conv2"](x)
        x = x + residual

        return x

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input images [batch, height, width, channels]

        Returns:
            features: [batch, features_dim]
        """
        # Normalize pixel values
        x = x.astype(jnp.float32) / 255.0

        # Process through blocks
        for block in self.blocks:
            x = self._forward_block(x, block)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Linear + ReLU
        x = nnx.relu(self.linear(x))

        return x


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
        hidden_sizes: List of hidden layer sizes (can be empty)
        output_size: Output dimension
        output_scale: Orthogonal initialization scale for output layer
        activation_fn: Activation function
        rngs: Random number generators

    Returns:
        Sequential network (or just output layer if hidden_sizes is empty)
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
    """Get activation function by name."""
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


# ---------------------------
# Networks
# ---------------------------
class ActorCriticCNN(nnx.Module):
    """Actor-critic network with CNN feature extractor for visual observations.

    Architecture:
    - Shared CNN feature extractor (NatureCNN or ImpalaCNN)
    - Separate policy and value heads (MLPs) on top of CNN features

    Supports both discrete and continuous action spaces.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_space: gym.Space,
        cnn_type: str,
        features_dim: int,
        actor_hidden_sizes: list[int],
        critic_hidden_sizes: list[int],
        actor_activation_fn: callable,
        critic_activation_fn: callable,
        rngs: nnx.Rngs,
    ):
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        # Build CNN feature extractor
        if cnn_type == "nature":
            self.features_extractor = NatureCNN(features_dim=features_dim, rngs=rngs)
        elif cnn_type == "impala":
            self.features_extractor = ImpalaCNN(features_dim=features_dim, rngs=rngs)
        else:
            raise ValueError(f"Unknown CNN type: {cnn_type}. Use 'nature' or 'impala'")

        # Determine output dimensions
        if self.is_discrete:
            action_dim = action_space.n
        else:
            action_dim = int(np.prod(action_space.shape))
            # Learnable log standard deviation for continuous actions
            self.action_logstd = nnx.Param(jnp.zeros(action_dim))

        # Build policy head (MLP on top of CNN features)
        self.policy_net = _build_mlp(
            features_dim,
            actor_hidden_sizes,
            action_dim,
            output_scale=0.01,
            activation_fn=actor_activation_fn,
            rngs=rngs,
        )

        # Build value head (separate, matching SB3)
        self.value_net = _build_mlp(
            features_dim,
            critic_hidden_sizes,
            1,
            output_scale=1.0,
            activation_fn=critic_activation_fn,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[distrax.Distribution, jax.Array]:
        """Forward pass returning action distribution and value estimate.

        Args:
            x: Visual observations [batch, height, width, channels]
               Expected: [batch, 84, 84, 4] for Atari

        Returns:
            (action_distribution, value_estimate)
        """
        # Extract features from CNN
        features = self.features_extractor(x)

        # Get policy output
        policy_output = self.policy_net(features)
        value = self.value_net(features).squeeze(-1)

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
            x: Visual observations [batch, height, width, channels]

        Returns:
            value_estimate: [batch]
        """
        features = self.features_extractor(x)
        return self.value_net(features).squeeze(-1)

    def get_action_and_value(
        self,
        x: jax.Array,
        action: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Get action with its log probability, entropy, and value.

        Args:
            x: Visual observations [batch, height, width, channels]
            action: Optional action to evaluate. If None, samples from policy.
            key: Random key for sampling (required if action is None)

        Returns:
            action, log_prob, entropy, value
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
    model: ActorCriticCNN,
    obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Jitted function for sampling actions during rollout collection."""
    dist, value = model(obs)
    action = dist.sample(seed=key)
    log_prob = dist.log_prob(action)
    return action, log_prob, value


@jax.jit
def predict_value(model: ActorCriticCNN, obs: jax.Array) -> jax.Array:
    """Jitted function for computing values (used for bootstrapping)."""
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
    """Compute Generalized Advantage Estimation using jax.lax.scan."""
    # Prepare data for backward scan
    values_with_next = jnp.concatenate([values, next_value[None, :]], axis=0)
    dones_with_next = jnp.concatenate([dones, next_done[None, :]], axis=0)

    def scan_fn(gae, t):
        """Scan function for GAE computation (processes timesteps backwards)."""
        reward = rewards[t]
        value = values[t]
        next_value_t = values_with_next[t + 1]
        next_nonterminal = 1.0 - dones_with_next[t + 1]

        # TD error
        delta = reward + gamma * next_value_t * next_nonterminal - value

        # GAE
        gae = delta + gamma * gae_lambda * next_nonterminal * gae

        return gae, gae

    # Scan backwards through time
    num_steps = rewards.shape[0]
    _, advantages = jax.lax.scan(
        scan_fn,
        init=jnp.zeros_like(next_value),
        xs=jnp.arange(num_steps - 1, -1, -1),
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
    model: ActorCriticCNN,
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

    # Value loss
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
    model: ActorCriticCNN,
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

    # Compute gradient norm for logging
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    info["loss/grad_norm"] = grad_norm

    # Update parameters
    optimizer.update(model=model, grads=grads)

    return info


# ---------------------------
# Checkpoint utilities
# ---------------------------
def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: ActorCriticCNN,
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
    model: ActorCriticCNN,
    optimizer: nnx.Optimizer,
) -> tuple[int, int, jax.Array]:
    """Load a checkpoint and restore model and optimizer state."""
    checkpoint_manager = ocp.CheckpointManager(
        directory=str(Path(checkpoint_path).resolve()),
        options=ocp.CheckpointManagerOptions(create=False),
    )

    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError(f"No checkpoints found in {checkpoint_path}")

    template_state = nnx.state(model)

    target_checkpoint = {
        "model_state": template_state,
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
def train(cfg: PPOVisionConfig):
    """Main PPO training loop for vision-based observations."""
    # Set random seeds
    np.random.seed(cfg.seed)
    key = jax.random.key(cfg.seed)

    # Create vectorized environment
    envs = gym.make_vec(
        cfg.env_id,
        num_envs=cfg.num_envs,
        vectorization_mode="sync",
    )

    # Seed the environments
    envs.reset(seed=cfg.seed)

    # Apply wrappers
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    # Apply frame stacking for temporal information
    if cfg.frame_stack > 1:
        # Note: Gymnasium's FrameStackObservation expects individual envs
        # For vectorized envs, we need to handle this differently
        # For simplicity, assuming the environment already provides stacked frames
        # or using a custom wrapper
        pass

    if isinstance(envs.single_action_space, gym.spaces.Box):
        envs = gym.wrappers.vector.ClipAction(envs)

    if cfg.norm_reward:
        envs = gym.wrappers.vector.NormalizeReward(envs, gamma=cfg.gamma)
        envs = gym.wrappers.vector.ClipReward(
            envs, min_reward=-cfg.clip_reward, max_reward=cfg.clip_reward
        )

    # Initialize model
    obs_shape = envs.single_observation_space.shape
    action_space = envs.single_action_space

    print(f"Environment: {cfg.env_id}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action space: {action_space}")

    # Get activation functions
    actor_activation_fn = get_activation_fn(cfg.actor_activation)
    critic_activation_fn = get_activation_fn(cfg.critic_activation)

    key, model_key = jax.random.split(key)
    model = ActorCriticCNN(
        obs_shape=obs_shape,
        action_space=action_space,
        cnn_type=cfg.cnn_type,
        features_dim=cfg.features_dim,
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
                alpha=0.0,
            )
        elif cfg.lr_schedule_type == "constant":
            lr_schedule = optax.constant_schedule(cfg.learning_rate)
        else:
            raise ValueError(f"Unknown lr_schedule_type: {cfg.lr_schedule_type}")
    else:
        lr_schedule = optax.constant_schedule(cfg.learning_rate)

    # Initialize optimizer
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
    env_name_clean = re.sub(r"-v\d+$", "", cfg.env_id.lower().replace("/", "-"))
    run_id = f"{time_stamp}-{env_name_clean}-ppo-vision-{cfg.seed}"

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

    # Training loop
    obs, _ = envs.reset(seed=cfg.seed)
    global_step = 0
    num_updates = 0

    # Resume from checkpoint if specified
    if cfg.resume_from is not None:
        global_step, num_updates, key = load_checkpoint(
            cfg.resume_from, model, optimizer
        )

    # Storage for rollouts (NumPy for CPU-side data)
    next_done = np.zeros(cfg.num_envs, dtype=np.float32)

    rollout_obs = np.zeros((cfg.num_steps, cfg.num_envs) + obs_shape, dtype=np.uint8)
    rollout_actions = np.zeros(
        (cfg.num_steps, cfg.num_envs) + action_space.shape, dtype=np.float32
    )
    rollout_rewards = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.float32)
    rollout_dones = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.float32)

    # JAX arrays for GPU-side data
    rollout_logprobs = jnp.zeros((cfg.num_steps, cfg.num_envs), dtype=jnp.float32)
    rollout_values = jnp.zeros((cfg.num_steps, cfg.num_envs), dtype=jnp.float32)

    # Episode tracking
    episode_returns = []
    episode_lengths = []

    start_time = time.time()

    for iteration in range(cfg.num_iterations):
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs

            # Store observations
            rollout_obs[step] = obs
            rollout_dones[step] = next_done

            # Sample action (pass uint8 observations, conversion happens in model)
            key, action_key = jax.random.split(key)
            action, logprob, value = predict_action_and_value(model, obs, action_key)

            # Store action (convert to NumPy for env.step)
            action_np = np.array(action)
            rollout_actions[step] = action_np

            # Store logprobs and values (stay on GPU)
            rollout_logprobs = rollout_logprobs.at[step].set(logprob)
            rollout_values = rollout_values.at[step].set(value)

            # Step environment
            obs, reward, terminated, truncated, info = envs.step(action_np)
            next_done = terminated | truncated

            # Store rewards
            rollout_rewards[step] = reward

            # Log episode stats
            if "_episode" in info and info["_episode"].any():
                episode_mask = info["_episode"]
                if "episode" in info:
                    episode_data = info["episode"]
                    for idx in np.where(episode_mask)[0]:
                        episode_returns.append(float(episode_data["r"][idx]))
                        episode_lengths.append(int(episode_data["l"][idx]))
            elif "final_info" in info:
                for env_info in info["final_info"]:
                    if env_info is not None and "episode" in env_info:
                        episode_returns.append(env_info["episode"]["r"])
                        episode_lengths.append(env_info["episode"]["l"])

        # Transfer NumPy arrays to GPU
        rollout_actions_jax = jnp.array(rollout_actions)
        rollout_obs_jax = jnp.array(rollout_obs)
        rollout_rewards_jax = jnp.array(rollout_rewards)
        rollout_dones_jax = jnp.array(rollout_dones)
        next_done_jax = jnp.array(next_done)

        # Bootstrap value for GAE
        next_value = predict_value(model, obs)

        # Compute advantages and returns
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

        # Update policy
        update_info = {}
        for epoch in range(cfg.update_epochs):
            # Random permutation for minibatches
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, cfg.rollout_buffer_size)

            for start in range(0, cfg.rollout_buffer_size, cfg.batch_size):
                mb_inds = perm[start : start + cfg.batch_size]

                # Normalize advantages per minibatch
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

                # Accumulate info
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

        # Average update info
        num_minibatch_updates = (epoch + 1) * cfg.num_minibatches
        for k in update_info:
            update_info[k] = float(update_info[k] / num_minibatch_updates)

        # Logging
        if num_updates % cfg.log_frequency == 0:
            sps = int(global_step / (time.time() - start_time))

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
    print(f"Algorithm: \t\tPPO (Vision/CNN)")
    print(f"CNN Architecture: \t{cfg.cnn_type.upper()}")
    print(f"Random Seed: \t\t{cfg.seed}")
    print(f"# envs: \t\t{cfg.num_envs}")
    print(f"# timesteps: \t\t{cfg.total_timesteps}")
    print(f"Logging directory: \t{cfg.checkpoint_dir}")
    print()


# ---------------------------
# Hydra entry point
# ---------------------------
@hydra.main(
    version_base=None, config_path="../../configs", config_name="ppo_gymnasium_vision"
)
def main(cfg: DictConfig):
    """Main entry point."""
    # Convert OmegaConf to dataclass
    ppo_cfg = PPOVisionConfig(**OmegaConf.to_container(cfg, resolve=True))
    huzzah(ppo_cfg)
    train(ppo_cfg)


if __name__ == "__main__":
    main()
