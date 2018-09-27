import gym

gym.envs.register(
    id='GridWorldEnv-v0',
    entry_point='environment.grid_world_env:GridWorld',
)