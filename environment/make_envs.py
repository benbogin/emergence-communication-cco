import gym


def make_env(env_id):
    def _thunk():
        env = gym.make(env_id)
        return env

    return _thunk