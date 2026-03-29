import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*warp.context.*")
warnings.filterwarnings("ignore", message=".*warp.math.*")

import hydra
from omegaconf import DictConfig, OmegaConf

from okapi.gymnasium.ppo import PPOConfig, train


def huzzah(cfg):
    OKAPI_LOGO = r"""                                                                 
       JJJJJJJJJJJJJJJJJJJJ                                                       
   JJJ::JJJ::::::::::::JJJ::JJJ                                                   
    J:::::::JJ:JJJJ:JJ:::::::J              JJJ                             JJJ   
    JJJ:::::::J::::J::J::::JJJ              JJJ                             JJJ   
    JJJJJJ::J::::::::J::JJJJJJ              JJJ                                   
    ::::::::J:::::::::J:::::::   JJJJJJJJ   JJJ JJJJ   JJJJJJJ   JJJJJJJJ   JJJ   
    :::::::J::J::::JJ:J:::::::  JJJ   JJJ   JJJJJJ          JJ   JJJ   JJJ  JJJ   
    JJJJJJJJ::::::::::JJJJJJJJ  JJJ    JJJ  JJJJJJ    JJJJJJJJ   JJJ   JJJ  JJJ   
    JJJJJJJJf:::::::::JJJJJJJJ  JJJ   JJJ   JJJ JJJ   JJJ   JJ   JJJ   JJJ  JJJ   
    JJJJJJJJJ::::::::JJJJJJJJJ    JJJJJJ    JJJ  JJJJ  JJJJJJJ   JJJJJJJJ   JJJ   
    :::::::::J::::::J:::::::::                                   JJJ              
     JJJJJJJJJJ:::::JJJJJJJJJ                                    JJJ              
      JJJJJJJJJ::::JJJJJJJJJ                                                      
                JJJ                                                                                                                                                                 
    """
    print(OKAPI_LOGO)
    print("\n" + "=" * 54)
    print(f"  Algorithm: \t\tPPO")
    print(f"  Environment: \t\t{cfg.env_id}")
    print(f"  # envs: \t\t{cfg.num_envs:,}")
    print(f"  # timesteps: \t\t{cfg.total_timesteps:,}")
    print(f"  # epochs: \t\t{cfg.num_iterations:,}")
    print(f"  Random Seed: \t\t{cfg.seed:,}")
    print("=" * 54 + "\n")


# ---------------------------
# Hydra entry point
# ---------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="ppo_gymnasium")
def main(cfg: DictConfig):
    ppo_cfg = PPOConfig(**OmegaConf.to_container(cfg, resolve=True))
    huzzah(ppo_cfg)
    train(ppo_cfg)


if __name__ == "__main__":
    main()
