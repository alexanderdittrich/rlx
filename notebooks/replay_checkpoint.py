#!/usr/bin/env python3
"""
Replay script for trained PPO models from checkpoints.

This script loads a trained model from a checkpoint and replays it in the environment,
optionally with rendering for visual evaluation.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from rlx.ppo import ActorCritic, load_checkpoint


def replay_checkpoint(
    checkpoint_path: str,
    env_id: str = None,
    num_episodes: int = 5,
    render: bool = True,
    seed: int = 42,
    deterministic: bool = False,
    save_video: bool = False,
    video_folder: str = "videos",
    max_episode_steps: int = None,
):
    """
    Replay a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        env_id: Environment ID (if None, tries to infer from checkpoint path)
        num_episodes: Number of episodes to replay
        render: Whether to render the environment
        seed: Random seed for reproducibility
        deterministic: If True, uses deterministic actions (no sampling)
        save_video: If True, saves videos of episodes
        video_folder: Folder to save videos
        max_episode_steps: Maximum steps per episode (None for env default)
    """
    print(f"🎬 Replaying checkpoint: {checkpoint_path}")

    # Infer environment ID from checkpoint path if not provided
    if env_id is None:
        path_parts = Path(checkpoint_path).name.split("-")
        if len(path_parts) >= 2:
            env_id = "-".join(path_parts[:-1])  # Remove seed suffix
        else:
            raise ValueError(
                "Could not infer environment ID from checkpoint path. Please specify env_id."
            )

    print(f"🎮 Environment: {env_id}")
    print(f"📊 Episodes: {num_episodes}")
    print(f"🎥 Rendering: {'Yes' if render else 'No'}")
    print(f"💾 Save video: {'Yes' if save_video else 'No'}")

    # Create environment
    env_kwargs = {}
    if render:
        env_kwargs["render_mode"] = "human"
    elif save_video:
        env_kwargs["render_mode"] = "rgb_array"

    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps

    env = gym.make(env_id, **env_kwargs)

    # Wrap for video recording if requested
    if save_video:
        video_path = Path(video_folder)
        video_path.mkdir(exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_path),
            name_prefix=f"replay_{env_id}",
            episode_trigger=lambda x: True,  # Record all episodes
        )

    print(f"📐 Observation space: {env.observation_space}")
    print(f"🎯 Action space: {env.action_space}")

    # Initialize model
    obs_shape = env.observation_space.shape
    action_space = env.action_space

    key = jax.random.key(seed)
    model = ActorCritic(
        obs_shape=obs_shape,
        action_space=action_space,
        hidden_sizes=[64, 64],  # Standard architecture
        rngs=nnx.Rngs(key),
    )

    # Create dummy optimizer for loading
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=3e-4), wrt=nnx.Param)

    # Load checkpoint
    try:
        global_step, num_updates, loaded_key = load_checkpoint(
            checkpoint_path, model, optimizer
        )
        print(f"✅ Loaded checkpoint:")
        print(f"   - Global step: {global_step}")
        print(f"   - Updates: {num_updates}")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        env.close()
        return None

    # Replay episodes
    episode_returns = []
    episode_lengths = []

    print(f"\n{'=' * 60}")
    print("🚀 Starting replay...")
    print(f"{'=' * 60}")

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_return = 0
        episode_length = 0
        done = False

        print(f"\n🎬 Episode {episode + 1}/{num_episodes}")

        while not done:
            # Get action from model
            obs_jax = jnp.array(obs).reshape(1, -1)

            if deterministic:
                # Use mean action for continuous, mode for discrete
                dist, _ = model(obs_jax)
                if model.is_discrete:
                    action = jnp.argmax(dist.logits, axis=-1)
                else:
                    action = dist.loc  # Mean of the distribution
            else:
                # Sample action
                action, _, _, _ = model.get_action_and_value(obs_jax, key=loaded_key)

            # Step environment
            # Convert action to appropriate format for the environment
            if model.is_discrete:
                action_for_env = int(action[0])  # Extract scalar for discrete actions
            else:
                action_for_env = np.array(action).flatten()  # Keep array for continuous

            obs, reward, terminated, truncated, info = env.step(action_for_env)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            # Optional: add small delay for better visualization
            if render and not save_video:
                time.sleep(0.01)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        print(f"   📊 Return: {episode_return:.1f}, Length: {episode_length}")

        # Show running statistics
        avg_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        print(f"   📈 Running avg: {avg_return:.2f} ± {std_return:.2f}")

    # Final statistics
    print(f"\n{'=' * 60}")
    print("📊 REPLAY SUMMARY")
    print(f"{'=' * 60}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {num_episodes}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Global step: {global_step}")
    print(f"Updates: {num_updates}")
    print(f"\nPerformance:")
    print(
        f"  Average Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}"
    )
    print(
        f"  Min/Max Return: {np.min(episode_returns):.1f} / {np.max(episode_returns):.1f}"
    )
    print(
        f"  Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )

    # Environment-specific success metrics
    if "CartPole" in env_id:
        success_threshold = 195
        success_rate = (
            sum(1 for r in episode_returns if r >= success_threshold)
            / len(episode_returns)
            * 100
        )
        print(f"  Success Rate (≥{success_threshold}): {success_rate:.1f}%")
    elif "LunarLander" in env_id:
        success_threshold = 200
        success_rate = (
            sum(1 for r in episode_returns if r >= success_threshold)
            / len(episode_returns)
            * 100
        )
        print(f"  Success Rate (≥{success_threshold}): {success_rate:.1f}%")

    if save_video:
        print(f"\n🎥 Videos saved to: {video_path}")

    env.close()

    return {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "checkpoint_info": {
            "global_step": global_step,
            "num_updates": num_updates,
            "checkpoint_path": checkpoint_path,
        },
    }


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Replay trained PPO models from checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("checkpoint_path", help="Path to checkpoint directory")

    parser.add_argument(
        "--env-id", help="Environment ID (auto-detected if not specified)"
    )

    parser.add_argument(
        "--episodes", "-n", type=int, default=5, help="Number of episodes to replay"
    )

    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (no sampling)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--save-video", action="store_true", help="Save video recordings of episodes"
    )

    parser.add_argument(
        "--video-folder", default="videos", help="Folder to save videos"
    )

    parser.add_argument("--max-steps", type=int, help="Maximum steps per episode")

    args = parser.parse_args()

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        return 1

    try:
        results = replay_checkpoint(
            checkpoint_path=str(checkpoint_path),
            env_id=args.env_id,
            num_episodes=args.episodes,
            render=not args.no_render,
            seed=args.seed,
            deterministic=args.deterministic,
            save_video=args.save_video,
            video_folder=args.video_folder,
            max_episode_steps=args.max_steps,
        )

        if results is None:
            return 1

        print("\n✅ Replay completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Replay interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error during replay: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
