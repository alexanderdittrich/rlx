import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*warp.context.*")
warnings.filterwarnings("ignore", message=".*warp.math.*")

import hydra
from omegaconf import DictConfig, OmegaConf

from rlx.playground.ppo import PPOConfig, train


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
    print("\n" + "=" * 54)
    print(f"  Algorithm: \t\tPPO")
    print(f"  Environment: \t\t{cfg.env_id}")
    print(f"  # envs: \t\t{cfg.num_envs:,}")
    print(f"  # timesteps: \t\t{cfg.total_timesteps:,}")
    print(f"  # epochs: \t\t{cfg.num_iterations:,}")
    print(f"  Random Seed: \t\t{cfg.seed:,}")
    print("=" * 54 + "\n")


# ---------------------------
# Hydra Entry Point
# ---------------------------
@hydra.main(config_path="../configs", config_name="rlx_ppo", version_base="1.3")
def main(cfg: DictConfig):
    """Main entry point with Hydra config."""
    # Convert OmegaConf to dataclass
    ppo_cfg = PPOConfig(**OmegaConf.to_container(cfg, resolve=True))
    huzzah(ppo_cfg)
    train(ppo_cfg)


if __name__ == "__main__":
    main()
