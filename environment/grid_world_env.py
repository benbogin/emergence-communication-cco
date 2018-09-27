import gym
import numpy as np
from gym.spaces import Discrete

from config import args
from environment.action import Action
from environment.objects.agent import Agent

from utils.render import merge_views_vertical, merge_views_horizontal

SCALE = 20

random_language = {}
random_language_set = set({})


class GridWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self):
        self.world = None
        self.inventory = None

        self.action_space = Discrete(len(Action))

        self.observation_space = (
            (args.world_size * args.cell_size, args.world_size * args.cell_size, 3),  # observation
            (args.world_size * args.cell_size, args.cell_size, 3)   # inventories
        )

        self.agent = Agent(color=(0, 255, 0))

        self.current_game = 0
        self.current_step = 0
        self.total_episodes = 0

        self.task_seed = None
        self.task_manager = None

        self.first_step_observation = None

    def _seed(self, seed=None):
        self.task_seed = seed

    def _reset(self):
        self.task = self.task_manager.get_task_from_seed(self.task_seed, self.agent)
        self.world, self.inventory = self.task.initialize_world_with_new_task()
        self.agent.world = self.world

        self.current_step = 0
        self.current_game += 1

        return self.state()

    def _step(self, action):
        self.world.agent_act(self.agent, Action(action))

        success = self.task.is_success()
        fail = self.task.is_fail()
        bonus = self.task.get_bonus()

        done = success or fail

        reward = 1.0 if success else -0.1 + bonus

        if self.current_step >= 30:
            done = True

        if done:
            self.total_episodes += 1
            self.task_manager.report(self.task_seed, success)

        self.current_step += 1

        return self.state(), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            return

        agents_views = self.world.render(pov=self.agent, scale=SCALE, borders=True)
        agents_inventory_views = self.agent.inventory.render(scale=SCALE, borders=True)
        inventory_views = self.inventory.render(scale=SCALE, borders=True)
        god_view = self.world.render(pov=None, scale=SCALE, borders=True)

        screen = merge_views_vertical((merge_views_horizontal((agents_views, god_view), padding=20),
                                       merge_views_horizontal((agents_inventory_views, inventory_views), padding=20)), padding=10)

        clipped = screen[:screen.shape[0] // 2 * 2, :screen.shape[1] // 2 * 2, :].astype(np.uint8)

        return clipped

    def state(self):
        # num_agents x 2 (current/goal) x size x size x 3 (colors)
        # note we want to hide the speaker from the listener
        return (
            np.stack([
                (255.0 - self.world.render(pov=self.agent, flip_horizontal=self.task.horizontal_flipped)) / 255 - 0.5,
                (255.0 - self.world.render(pov=self.agent)) / 255 - 0.5,
            ]),
            ((255.0 - np.stack((
                self.agent.inventory.render(),
                self.inventory.render()
            ))) / 255 - 0.5)
        )
