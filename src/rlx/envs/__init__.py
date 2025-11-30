"""Environment registrations for rlx."""

from gymnasium.envs.registration import register

# Register HalfCheetahDir environment
register(
    id='HalfCheetahDir-v0',
    entry_point='rlx.envs.gymnasium.cheetahdir:HalfCheetahDirEnv',
    max_episode_steps=1000,
)

# Register AntDir environment
register(
    id='AntDir-v0',
    entry_point='rlx.envs.gymnasium.antdir:AntDirEnv',
    max_episode_steps=1000,
)
