import numpy as np


class MultiVecEnv:
    def __init__(self, env_fns, task_manager):
        self.envs = [env_fn() for env_fn in env_fns]
        for env in self.envs:
            env.unwrapped.task_manager = task_manager
        self.task_manager = task_manager
        self.action_space, self.observation_space = self.envs[0].action_space, self.envs[0].observation_space

    def step(self, actions):
        obs, invs, rewards, dones, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            (ob, inv), reward, done, info = env.step(action)
            if done:
                env.seed(self.task_manager.get_seed())
                ob, inv = env.reset()

            obs.append(ob)
            invs.append(inv)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return np.stack(obs), np.stack(invs), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        obs = []
        invs = []
        for env in self.envs:
            env.seed(self.task_manager.get_seed())
            worlds, inventory = env.reset()
            obs.append(worlds)
            invs.append(inventory)
        return np.stack(obs), np.stack(invs)

    def close(self):
        for env in self.envs:
            env.close()
