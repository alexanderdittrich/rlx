# PPO for Visual Observations (CNN-based)

This module provides a PPO implementation specifically designed for visual observations (image-based environments), such as Atari games. It follows the architectural choices from Stable-Baselines3.

## Features

- **CNN Feature Extractors**: Two architectures available
  - **NatureCNN**: Standard architecture from the DQN Nature paper (Mnih et al., 2015)
  - **ImpalaCNN**: Deeper architecture from the IMPALA paper for complex visual tasks

- **Flexible Architecture**: Separate policy and value networks with configurable hidden layers
- **Frame Stacking**: Support for temporal information through frame stacking
- **Atari Environments**: Optimized hyperparameters for Atari games
- **Checkpointing**: Save and resume training with Orbax

## Architecture Details

### NatureCNN (Default)

The standard architecture for 84x84 Atari observations with 4 stacked frames:

```
Input: [batch, 84, 84, 4]
  ↓
Conv1: 32 filters, 8x8 kernel, stride 4, ReLU
  ↓
Conv2: 64 filters, 4x4 kernel, stride 2, ReLU
  ↓
Conv3: 64 filters, 3x3 kernel, stride 1, ReLU
  ↓
Flatten → [batch, 3136]
  ↓
Linear: 512 units, ReLU
  ↓
Features: [batch, 512]
```

The 512-dimensional feature vector is then fed to:
- **Policy head**: Linear layer → action logits/mean
- **Value head**: Linear layer → value estimate

### ImpalaCNN (Alternative)

A deeper architecture with residual connections:

```
Input: [batch, 84, 84, 4]
  ↓
3x Residual Blocks (channels: 16 → 32 → 32)
  Each block:
    - Conv 3x3, stride 1
    - MaxPool 3x3, stride 2
    - 2x Residual layers (Conv-ReLU-Conv-ReLU + skip)
  ↓
Flatten
  ↓
Linear: 512 units, ReLU
  ↓
Features: [batch, 512]
```

## Installation

Requires gymnasium with Atari support:

```bash
pip install "gymnasium[atari,accept-rom-license]"
```

Or if you already have the ROMs:

```bash
pip install "gymnasium[atari]"
```

## Usage

### Training

Basic training with default settings (Pong):

```bash
python src/rlx/ppo_visual.py
```

Custom environment and settings:

```bash
python src/rlx/ppo_visual.py \
    env_id="ALE/Breakout-v5" \
    total_timesteps=20_000_000 \
    num_envs=16 \
    use_wandb=true
```

Using IMPALA CNN architecture:

```bash
python src/rlx/ppo_visual.py \
    cnn_type="impala" \
    features_dim=512
```

Resume from checkpoint:

```bash
python src/rlx/ppo_visual.py \
    resume_from="checkpoints/20250101_120000-pong-ppo-visual-42"
```

### Evaluation/Replay

Replay a trained model:

```bash
python notebooks/replay_visual.py \
    --checkpoint_dir checkpoints/20250101_120000-pong-ppo-visual-42 \
    --num_episodes 5 \
    --render
```

Record video:

```bash
python notebooks/replay_visual.py \
    --checkpoint_dir checkpoints/20250101_120000-pong-ppo-visual-42 \
    --num_episodes 3 \
    --record_video \
    --video_dir videos/pong
```

## Configuration

Key hyperparameters (following SB3 Atari defaults):

```yaml
# Environment
env_id: "ALE/Pong-v5"
num_envs: 8

# Training
total_timesteps: 10_000_000
num_steps: 128
batch_size: 256
update_epochs: 4

# PPO
learning_rate: 2.5e-4
clip_coef: 0.1  # Lower than continuous control (0.2)
gamma: 0.99
gae_lambda: 0.95
ent_coef: 0.01

# CNN
cnn_type: "nature"  # or "impala"
features_dim: 512
frame_stack: 4
```

## Performance Tips

### GPU Memory

Visual observations require more GPU memory than vector observations:
- Reduce `num_envs` if running out of memory
- Reduce `batch_size` if needed
- Use mixed precision training (future feature)

### Training Speed

- **Batch size**: Larger batches better utilize GPU parallelism
- **Number of environments**: More environments = more samples per update
- **Frame stack**: Default of 4 is standard and works well
- **CNN type**: NatureCNN is faster, ImpalaCNN may perform better on complex tasks

### Hyperparameter Tuning

For Atari games, the defaults are well-tuned, but you can adjust:
- `learning_rate`: Try 1e-4 to 5e-4
- `clip_coef`: 0.1 is standard for Atari (vs 0.2 for continuous)
- `ent_coef`: Increase (e.g., 0.02) for more exploration
- `update_epochs`: 3-4 is typical for Atari

## Differences from Standard PPO

1. **CNN Feature Extractor**: Replaces MLP encoder
2. **Frame Stacking**: 4 frames provide temporal information
3. **Clip Coefficient**: 0.1 instead of 0.2 for Atari
4. **Reward Normalization**: Enabled by default for Atari
5. **No Observation Normalization**: Visual obs are already in [0, 255]
6. **ReLU Activation**: Standard for visual tasks (vs tanh for continuous)

## Supported Environments

Any Gymnasium environment with visual observations (Box observation space with image format):

- **Atari**: All ALE environments (e.g., `ALE/Pong-v5`, `ALE/Breakout-v5`)
- **MuJoCo**: With `render_mode="rgb_array"` observations
- **Custom**: Any environment with image observations [H, W, C]

## Example: Training on Breakout

```bash
# Train for 20M timesteps with 16 parallel environments
python src/rlx/ppo_visual.py \
    env_id="ALE/Breakout-v5" \
    total_timesteps=20_000_000 \
    num_envs=16 \
    batch_size=512 \
    checkpoint_frequency=50 \
    use_wandb=true \
    wandb_project="atari-ppo"
```

Expected training time: ~2-4 hours on a modern GPU (RTX 3090 or similar)

## Architecture Comparison

| Feature | NatureCNN | ImpalaCNN |
|---------|-----------|-----------|
| Depth | 3 conv layers | 3 residual blocks (9 conv layers) |
| Parameters | ~1.7M | ~4.5M |
| Speed | Faster | Slower |
| Performance | Good for standard Atari | Better for complex visual tasks |
| Use case | Default choice | Complex environments |

## References

1. **NatureCNN**: Mnih et al. (2015) "Human-level control through deep reinforcement learning"
2. **ImpalaCNN**: Espeholt et al. (2018) "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
3. **PPO**: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
4. **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3

## Troubleshooting

### Import Errors

If you get `ImportError: cannot import name 'ActorCriticCNN'`:
- Make sure you're running from the workspace root
- Check that `src/rlx/ppo_visual.py` exists

### ROM Errors

If you get ROM-related errors:
```bash
pip install "gymnasium[accept-rom-license]"
```

### Out of Memory

Reduce memory usage:
```bash
python src/rlx/ppo_visual.py \
    num_envs=4 \
    batch_size=128
```

### Slow Training

Increase parallelism:
```bash
python src/rlx/ppo_visual.py \
    num_envs=16 \
    batch_size=512
```
