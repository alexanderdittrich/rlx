"""
VariBAD 
=========

Reproduction of VariBAD from https://github.com/lmzintgraf/varibad
All components in one file for the CheetahDir task.

Architecture:
1. RNNEncoder: (s,a,r) → posterior q(z|context)
2. Decoders: Reward, State, Task reconstruction
3. VAE: ELBO = reconstruction - KL
4. Policy: Actor-Critic conditioned on z
5. MetaLearner: Training loop across tasks
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import gymnasium as gym
from flax import nnx

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Register custom environments
import rlx.envs


@dataclass 
class VariBADConfig:
    """All hyperparameters matching original VariBAD.
    
    Supported environments:
    - HalfCheetahDir-v0: 17D obs, 6D action
    - AntDir-v0: 105D obs, 8D action
    """
    # Env
    env_name: str = "HalfCheetahDir-v0"
    num_envs: int = 32
    max_rollouts_per_task: int = 2
    
    # Training
    num_frames: int = int(2e7)
    policy_num_steps: int = 200
    
    # VAE architecture
    latent_dim: int = 5
    encoder_gru_hidden_size: int = 128
    encoder_layers_before_gru: List[int] = field(default_factory=list)
    encoder_layers_after_gru: List[int] = field(default_factory=list)
    action_embedding_size: int = 16
    state_embedding_size: int = 32
    reward_embedding_size: int = 16
    
    # Decoders
    decode_reward: bool = True
    decode_state: bool = False
    decode_task: bool = False
    reward_decoder_layers: List[int] = field(default_factory=lambda: [64, 32])
    state_decoder_layers: List[int] = field(default_factory=lambda: [64, 32])
    task_decoder_layers: List[int] = field(default_factory=lambda: [64, 32])
    
    # VAE training
    size_vae_buffer: int = 1000
    vae_batch_num_trajs: int = 25
    num_vae_updates: int = 3
    lr_vae: float = 0.001
    kl_weight: float = 1.0
    rew_loss_coeff: float = 1.0
    state_loss_coeff: float = 1.0
    task_loss_coeff: float = 1.0
    
    # Policy
    policy_layers: List[int] = field(default_factory=lambda: [128, 128])
    pass_state_to_policy: bool = True
    pass_latent_to_policy: bool = True
    policy_init_std: float = 1.0
    
    # PPO
    lr_policy: float = 0.0007
    policy_num_epochs: int = 2
    policy_num_minibatch: int = 16
    policy_gamma: float = 0.97
    policy_tau: float = 0.95
    policy_value_loss_coef: float = 0.5
    policy_entropy_coef: float = 0.01
    policy_max_grad_norm: float = 0.5
    policy_clip_param: float = 0.2
    
    # Misc
    seed: int = 73
    log_interval: int = 25
    
    # Logging & Visualization
    use_wandb: bool = False
    wandb_project: str = "varibad-rlx"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    verbose: bool = True


class RolloutStorageVAE:
    """Stores trajectories for VAE training."""
    def __init__(self, max_num_rollouts, state_dim, action_dim):
        self.prev_state = deque(maxlen=max_num_rollouts)
        self.next_state = deque(maxlen=max_num_rollouts)
        self.actions = deque(maxlen=max_num_rollouts)
        self.rewards = deque(maxlen=max_num_rollouts)
        self.trajectory_lens = deque(maxlen=max_num_rollouts)
    
    def insert_trajectory(self, prev_states, actions, next_states, rewards):
        self.prev_state.append(prev_states)
        self.next_state.append(next_states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.trajectory_lens.append(len(rewards))
    
    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.prev_state), min(batch_size, len(self.prev_state)), replace=False)
        max_len = max([self.trajectory_lens[i] for i in indices])
        
        # Get first trajectory to determine dimensions
        first_prev = self.prev_state[indices[0]]
        first_action = self.actions[indices[0]]
        
        prev_obs = jnp.zeros((max_len, len(indices), first_prev.shape[-1]))
        next_obs = jnp.zeros((max_len, len(indices), first_prev.shape[-1]))
        actions = jnp.zeros((max_len, len(indices), first_action.shape[-1]))
        rewards = jnp.zeros((max_len, len(indices), 1))
        
        for i, idx in enumerate(indices):
            length = self.trajectory_lens[idx]
            # Convert numpy arrays to jax arrays and handle shapes properly
            prev_data = jnp.array(self.prev_state[idx])
            next_data = jnp.array(self.next_state[idx])
            action_data = jnp.array(self.actions[idx])
            reward_data = jnp.array(self.rewards[idx])
            
            # Ensure correct shapes
            if prev_data.ndim == 3:  # [timesteps, envs, features]
                prev_data = prev_data[:, 0, :]  # Take first env
            if next_data.ndim == 3:
                next_data = next_data[:, 0, :]
            if action_data.ndim == 3:
                action_data = action_data[:, 0, :]
            if reward_data.ndim == 2:
                reward_data = reward_data[:, 0]
                
            prev_obs = prev_obs.at[:length, i].set(prev_data[:length])
            next_obs = next_obs.at[:length, i].set(next_data[:length])
            actions = actions.at[:length, i].set(action_data[:length])
            rewards = rewards.at[:length, i].set(reward_data[:length].reshape(-1, 1))
        
        return prev_obs, next_obs, actions, rewards


class RNNEncoder(nnx.Module):
    """Encodes (s,a,r) sequences into latent posterior q(z|τ)."""
    def __init__(self, args, state_dim, action_dim, rngs):
        # Embedding layers
        self.state_enc = nnx.Linear(state_dim, args.state_embedding_size, rngs=rngs)
        self.action_enc = nnx.Linear(action_dim, args.action_embedding_size, rngs=rngs)
        self.reward_enc = nnx.Linear(1, args.reward_embedding_size, rngs=rngs)
        
        embed_dim = args.state_embedding_size + args.action_embedding_size + args.reward_embedding_size
        self.gru = nnx.GRUCell(in_features=embed_dim, hidden_features=args.encoder_gru_hidden_size, rngs=rngs)
        
        self.fc_mu = nnx.Linear(args.encoder_gru_hidden_size, args.latent_dim, rngs=rngs)
        self.fc_logvar = nnx.Linear(args.encoder_gru_hidden_size, args.latent_dim, rngs=rngs)
        
        self.hidden_size = args.encoder_gru_hidden_size
        self.latent_dim = args.latent_dim
    
    def prior(self, batch_size):
        """Return prior N(0,I)."""
        return jnp.zeros((batch_size, self.latent_dim)), jnp.zeros((batch_size, self.latent_dim))
    
    def __call__(self, states, actions, rewards):
        """Encode sequence."""
        seq_len, batch_size = states.shape[0], states.shape[1]
        
        # Embed
        hs = nnx.relu(self.state_enc(states.reshape(-1, states.shape[-1]))).reshape(seq_len, batch_size, -1)
        ha = nnx.relu(self.action_enc(actions.reshape(-1, actions.shape[-1]))).reshape(seq_len, batch_size, -1)
        hr = nnx.relu(self.reward_enc(rewards.reshape(-1, 1))).reshape(seq_len, batch_size, -1)
        
        h = jnp.concatenate([hs, ha, hr], axis=-1)
        
        # GRU
        hidden = jnp.zeros((batch_size, self.hidden_size))
        for t in range(seq_len):
            hidden, _ = self.gru(hidden, h[t])  # GRUCell(carry, inputs) returns (new_carry, output)
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar


class RewardDecoder(nnx.Module):
    """Decodes r from (z, s, a, s')."""
    def __init__(self, layers, latent_dim, state_dim, state_embed_dim, action_dim, action_embed_dim, rngs):
        self.state_enc = nnx.Linear(state_dim, state_embed_dim, rngs=rngs)
        self.action_enc = nnx.Linear(action_dim, action_embed_dim, rngs=rngs)
        
        input_dim = latent_dim + 2 * state_embed_dim + action_embed_dim
        self.layers = []
        curr = input_dim
        for h in layers:
            self.layers.append(nnx.Linear(curr, h, rngs=rngs))
            curr = h
        self.fc_out = nnx.Linear(curr, 1, rngs=rngs)
    
    def __call__(self, latent, next_state, prev_state, action):
        hps = nnx.relu(self.state_enc(prev_state))
        hns = nnx.relu(self.state_enc(next_state))
        ha = nnx.relu(self.action_enc(action))
        
        h = jnp.concatenate([latent, hps, hns, ha], axis=-1)
        for layer in self.layers:
            h = nnx.relu(layer(h))
        return self.fc_out(h)


class Policy(nnx.Module):
    """Actor-Critic conditioned on latent z."""
    def __init__(self, args, obs_dim, action_space, rngs):
        input_dim = obs_dim + (args.latent_dim if args.pass_latent_to_policy else 0)
        
        # Actor
        self.actor_layers = []
        curr = input_dim
        for h in args.policy_layers:
            self.actor_layers.append(nnx.Linear(curr, h, rngs=rngs))
            curr = h
        
        # Critic
        self.critic_layers = []
        curr = input_dim
        for h in args.policy_layers:
            self.critic_layers.append(nnx.Linear(curr, h, rngs=rngs))
            curr = h
        self.critic_out = nnx.Linear(args.policy_layers[-1], 1, rngs=rngs)
        
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.is_discrete:
            action_dim = action_space.n
            self.actor_out = nnx.Linear(args.policy_layers[-1], action_dim, rngs=rngs)
        else:
            action_dim = action_space.shape[0]
            self.actor_mean = nnx.Linear(args.policy_layers[-1], action_dim, rngs=rngs)
            self.actor_logstd = nnx.Param(jnp.log(jnp.ones(action_dim) * args.policy_init_std))
    
    def __call__(self, state, latent):
        x = jnp.concatenate([state, latent], axis=-1)
        
        # Actor
        h = x
        for layer in self.actor_layers:
            h = nnx.tanh(layer(h))
        
        if self.is_discrete:
            dist = distrax.Categorical(logits=self.actor_out(h))
        else:
            mean = self.actor_mean(h)
            std = jnp.exp(self.actor_logstd.value)
            dist = distrax.MultivariateNormalDiag(mean, std)
        
        # Critic
        h = x
        for layer in self.critic_layers:
            h = nnx.tanh(layer(h))
        value = self.critic_out(h).squeeze(-1)
        
        return dist, value
    
    def act(self, state, latent):
        dist, value = self(state, latent)
        action = dist.sample(seed=jax.random.PRNGKey(0))
        return action, value


class VariBADVAE:
    """VAE for task inference."""
    def __init__(self, args, encoder, reward_decoder):
        self.args = args
        self.encoder = encoder
        self.reward_decoder = reward_decoder
        self.optimizer = nnx.Optimizer(encoder, optax.adam(args.lr_vae))
    
    def compute_loss(self, mu, logvar, prev_obs, next_obs, actions, rewards):
        # Sample latent
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(jax.random.PRNGKey(0), mu.shape)
        z = mu + std * eps
        
        # Reconstruct rewards
        z_expanded = jnp.repeat(z[:, None], prev_obs.shape[0], axis=1).reshape(-1, z.shape[-1])
        prev_flat = prev_obs.reshape(-1, prev_obs.shape[-1])
        next_flat = next_obs.reshape(-1, next_obs.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        
        pred_reward = self.reward_decoder(z_expanded, next_flat, prev_flat, act_flat)
        rew_loss = jnp.mean((pred_reward - rewards.reshape(-1, 1)) ** 2)
        
        # KL
        kl = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1).mean()
        
        total_loss = self.args.rew_loss_coeff * rew_loss + self.args.kl_weight * kl
        return total_loss, (rew_loss, kl)


class MetaLearner:
    """Main training loop."""
    def __init__(self, args):
        self.args = args
        
        # Create envs
        self.envs = gym.make_vec(args.env_name, num_envs=args.num_envs, vectorization_mode="sync")
        obs_shape = self.envs.single_observation_space.shape
        action_space = self.envs.single_action_space
        
        self.state_dim = obs_shape[0]
        self.action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else action_space.n
        
        # Build networks
        key = jax.random.PRNGKey(args.seed)
        key, enc_key, pol_key, dec_key = jax.random.split(key, 4)
        
        self.encoder = RNNEncoder(args, self.state_dim, self.action_dim, nnx.Rngs(enc_key))
        
        self.reward_decoder = None
        if args.decode_reward:
            self.reward_decoder = RewardDecoder(
                args.reward_decoder_layers, args.latent_dim,
                self.state_dim, args.state_embedding_size,
                self.action_dim, args.action_embedding_size, 
                nnx.Rngs(dec_key)
            )
        
        self.policy = Policy(args, self.state_dim, action_space, nnx.Rngs(pol_key))
        
        # VAE
        self.vae = VariBADVAE(args, self.encoder, self.reward_decoder)
        
        # Storage
        self.vae_storage = RolloutStorageVAE(args.size_vae_buffer, self.state_dim, self.action_dim)
        
        self.frames = 0
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.all_episode_rewards = []  # For plotting
        self.vae_losses = []
        self.start_time = time.time()
        
        # Wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or f"varibad-{args.env_name}-{args.seed}",
                config=vars(args)
            )
            print(f"✓ Wandb initialized: {wandb.run.name}")
    
    def train(self):
        if self.args.verbose:
            print("\n" + "="*60)
            print(f"Starting VariBAD training on {self.args.env_name}")
            print(f"Latent dim: {self.args.latent_dim}, Num envs: {self.args.num_envs}")
            print("="*60 + "\n")
        
        obs, _ = self.envs.reset(seed=self.args.seed)
        
        # Episode tracking
        episode_rewards_buffer = np.zeros(self.args.num_envs)
        episode_lengths_buffer = np.zeros(self.args.num_envs, dtype=int)
        
        # Collect trajectories
        for iteration in range(100):  # Simplified
            # Collect episode
            trajectory_obs = []
            trajectory_actions = []
            trajectory_rewards = []
            trajectory_next_obs = []
            
            iter_rewards = []
            
            for step in range(self.args.policy_num_steps):
                # Get latent (prior for now)
                mu, logvar = self.encoder.prior(self.args.num_envs)
                z = mu  # Use mean
                
                # Act
                action, value = self.policy.act(jnp.array(obs), z)
                
                next_obs, reward, term, trunc, info = self.envs.step(np.array(action))
                done = term | trunc
                
                # Track episodes
                episode_rewards_buffer += reward
                episode_lengths_buffer += 1
                
                for i in range(self.args.num_envs):
                    if done[i]:
                        self.episode_rewards.append(episode_rewards_buffer[i])
                        self.episode_lengths.append(episode_lengths_buffer[i])
                        self.all_episode_rewards.append(episode_rewards_buffer[i])
                        episode_rewards_buffer[i] = 0
                        episode_lengths_buffer[i] = 0
                
                trajectory_obs.append(obs)
                trajectory_actions.append(np.array(action))
                trajectory_rewards.append(reward)
                trajectory_next_obs.append(next_obs)
                iter_rewards.append(reward)
                
                obs = next_obs
                self.frames += self.args.num_envs
            
            # Store trajectory
            if len(trajectory_rewards) > 0:
                self.vae_storage.insert_trajectory(
                    np.array(trajectory_obs),
                    np.array(trajectory_actions),
                    np.array(trajectory_next_obs),
                    np.array(trajectory_rewards)
                )
            
            # Update VAE
            vae_loss_value = 0.0
            rew_loss_value = 0.0
            kl_value = 0.0
            if len(self.vae_storage.prev_state) >= self.args.vae_batch_num_trajs:
                for _ in range(self.args.num_vae_updates):
                    prev_obs, next_obs, actions, rewards = self.vae_storage.get_batch(self.args.vae_batch_num_trajs)
                    
                    mu, logvar = self.encoder(next_obs, actions, rewards)
                    
                    def loss_fn(encoder):
                        mu, logvar = encoder(next_obs, actions, rewards)
                        return self.vae.compute_loss(mu, logvar, prev_obs, next_obs, actions, rewards)
                    
                    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
                    (total_loss, (rew_loss, kl)), grads = grad_fn(self.encoder)
                    self.vae.optimizer.update(grads)
                    
                    vae_loss_value = float(total_loss)
                    rew_loss_value = float(rew_loss)
                    kl_value = float(kl)
                
                self.vae_losses.append(vae_loss_value)
            
            # Logging
            if iteration % self.args.log_interval == 0:
                elapsed = time.time() - self.start_time
                fps = self.frames / elapsed if elapsed > 0 else 0
                
                mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
                mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0.0
                iter_mean_reward = np.mean(iter_rewards) if iter_rewards else 0.0
                
                if self.args.verbose:
                    print(f"Iter {iteration:3d} | Frames {self.frames:7d} | FPS {fps:6.0f} | "
                          f"Mean Reward {mean_reward:7.2f} | Iter Reward {iter_mean_reward:7.2f} | "
                          f"VAE Loss {vae_loss_value:7.4f} | Rew Loss {rew_loss_value:7.4f} | KL {kl_value:7.4f}")
                
                # Wandb logging
                if self.args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "iteration": iteration,
                        "frames": self.frames,
                        "fps": fps,
                        "mean_reward_100": mean_reward,
                        "mean_episode_length": mean_length,
                        "iter_mean_reward": iter_mean_reward,
                        "vae_loss": vae_loss_value,
                        "reward_loss": rew_loss_value,
                        "kl_divergence": kl_value,
                        "num_episodes": len(self.episode_rewards),
                    })
        
        if self.args.verbose:
            print("\n" + "="*60)
            print("Training complete!")
            print(f"Total episodes: {len(self.episode_rewards)}")
            if self.episode_rewards:
                print(f"Final mean reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
            print("="*60 + "\n")
        
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        self.envs.close()


def main():
    cfg = VariBADConfig()
    cfg.num_frames = 1_000_000
    cfg.num_envs = 16
    cfg.policy_num_steps = 50
    cfg.verbose = True
    cfg.log_interval = 10
    cfg.use_wandb = True  # Uncomment to enable wandb logging
    
    meta_learner = MetaLearner(cfg)
    meta_learner.train()


if __name__ == "__main__":
    main()
