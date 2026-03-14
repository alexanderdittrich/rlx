"""
Replay trained PPO Vision policies

Load a trained model checkpoint and visualize the agent's behavior in the environment.
Supports rendering and optional video recording.

Usage:
    python replay_gymnasium_vision.py --checkpoint_dir checkpoints/20250101_120000-pong-ppo-vision-42 \\
                                      --num_episodes 5 --render
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


from rlx.gymnasium.ppo_vision import ActorCriticCNN, PPOVisionConfig, get_activation_fn
import orbax.checkpoint as ocp
from omegaconf import OmegaConf


def load_model_from_checkpoint(checkpoint_dir: str):
    """Load model and config from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        model: Loaded ActorCriticCNN model
        config: PPOVisionConfig
    """
    checkpoint_path = Path(checkpoint_dir).resolve()

    # Load config from checkpoint directory (if saved)
    config_path = checkpoint_path / ".hydra" / "config.yaml"
    if config_path.exists():
        cfg_dict = OmegaConf.load(config_path)
        cfg = PPOVisionConfig(**OmegaConf.to_container(cfg_dict, resolve=True))
    else:
        print("Warning: No config found in checkpoint, using defaults")
        cfg = PPOVisionConfig()

    # Create environment to get observation/action spaces
    env = gym.make(cfg.env_id)
    obs_shape = env.observation_space.shape
    action_space = env.action_space
    env.close()

    # Initialize model with same architecture
    actor_activation_fn = get_activation_fn(cfg.actor_activation)
    critic_activation_fn = get_activation_fn(cfg.critic_activation)

    model = ActorCriticCNN(
        obs_shape=obs_shape,
        action_space=action_space,
        cnn_type=cfg.cnn_type,
        features_dim=cfg.features_dim,
        actor_hidden_sizes=cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        actor_activation_fn=actor_activation_fn,
        critic_activation_fn=critic_activation_fn,
        rngs=nnx.Rngs(0),  # Dummy RNG for initialization
    )

    # Load checkpoint
    checkpoint_manager = ocp.CheckpointManager(
        directory=str(checkpoint_path),
        options=ocp.CheckpointManagerOptions(create=False),
    )

    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError(f"No checkpoints found in {checkpoint_path}")

    # Create target structure for restoration
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

    # Restore model state
    nnx.update(model, checkpoint_state["model_state"])

    print(f"Loaded checkpoint from step {latest_step}")
    print(f"  Global step: {checkpoint_state['global_step']}")
    print(f"  Updates: {checkpoint_state['num_updates']}")

    return model, cfg


@jax.jit
def get_action(model: ActorCriticCNN, obs: jax.Array, deterministic: bool = True):
    """Get action from model (deterministic or stochastic).

    Args:
        model: Trained model
        obs: Observation [1, H, W, C]
        deterministic: If True, use mode/mean of distribution. If False, sample.

    Returns:
        action: Selected action
    """
    dist, _ = model(obs)

    if deterministic:
        # For discrete: argmax of logits
        # For continuous: mean of distribution
        if hasattr(dist, "logits"):
            # Categorical distribution (discrete actions)
            action = jnp.argmax(dist.logits, axis=-1)
        else:
            # Continuous distribution
            action = dist.mode()
    else:
        # Sample from distribution
        action = dist.sample(seed=jax.random.key(0))

    return action


def replay(
    checkpoint_dir: str,
    num_episodes: int = 5,
    render: bool = True,
    deterministic: bool = True,
    record_video: bool = False,
    video_dir: str = "videos",
):
    """Replay trained policy.

    Args:
        checkpoint_dir: Path to checkpoint directory
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
        record_video: Whether to record video
        video_dir: Directory to save videos
    """
    # Load model and config
    model, cfg = load_model_from_checkpoint(checkpoint_dir)

    # Create environment
    render_mode = "rgb_array" if record_video else ("human" if render else None)
    env = gym.make(cfg.env_id, render_mode=render_mode)

    # Wrap for video recording if requested
    if record_video:
        video_path = Path(video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_path),
            episode_trigger=lambda x: True,  # Record all episodes
        )

    # Apply frame stacking if needed (same as training)
    if cfg.frame_stack > 1:
        env = gym.wrappers.FrameStack(env, num_stack=cfg.frame_stack)

    episode_returns = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            # Add batch dimension and convert to JAX array
            obs_jax = jnp.array(obs)[None, ...]

            # Get action
            action = get_action(model, obs_jax, deterministic=deterministic)
            action = np.array(action).squeeze()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            if render and not record_video:
                env.render()

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        print(
            f"Episode {episode + 1}/{num_episodes}: "
            f"Return = {episode_return:.2f}, Length = {episode_length}"
        )

    env.close()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics:")
    print(
        f"  Mean Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}"
    )
    print(
        f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(f"  Min Return: {np.min(episode_returns):.2f}")
    print(f"  Max Return: {np.max(episode_returns):.2f}")
    print("=" * 50)

    if record_video:
        print(f"\nVideos saved to: {video_dir}")


def main():
    parser = argparse.ArgumentParser(description="Replay trained PPO Vision policy")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during replay",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record video of episodes",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="Directory to save videos",
    )

    args = parser.parse_args()

    replay(
        checkpoint_dir=args.checkpoint_dir,
        num_episodes=args.num_episodes,
        render=args.render,
        deterministic=not args.stochastic,
        record_video=args.record_video,
        video_dir=args.video_dir,
    )


if __name__ == "__main__":
    main()
