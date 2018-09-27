from collections import Counter, OrderedDict
import numpy as np
from config import args
from environment.tasks.missions import Missions
from environment.tasks.color_number import ColorNumberQuantifier
from environment.tasks.task import Task

mission_types = OrderedDict([
    ('Collect', Missions.Collect),
    ('Colorize', Missions.Colorize),
    ('Bring', Missions.Bring),
])


class TaskManager:
    def __init__(self, n_missions, unseen_tasks=None, total_tasks=200000, test_mode=False):
        self.total_tasks = total_tasks

        task_start = 1000000 if test_mode else 0
        task_end = task_start + total_tasks

        self.seeds = np.arange(task_start, task_end).tolist()
        self.n_missions = n_missions

        self.unseen_tasks = unseen_tasks if unseen_tasks is not None else []

        self.task_seed = -1

        self.success_per_task = Counter()
        self.total_per_task = Counter()
        self.success_pc_per_task = {}

        self.test_mode = test_mode

    def get_seed(self):
        np.random.seed(None)

        self.task_seed += 1

        if self.task_seed == len(self.seeds) -1:
            np.random.shuffle(self.seeds)
            self.task_seed = -1

        return self.seeds[self.task_seed]

    def reset_report(self):
        self.success_per_task = Counter()
        self.total_per_task = Counter()

    def report_full(self):
        return len(self.total_per_task) == self.total_tasks

    def get_success_percentage(self):
        assert self.report_full()
        return sum(self.success_per_task.values()) / self.total_tasks

    def get_unsuccessful_tasks(self):
        return set(self.total_per_task.keys()) - set(self.success_per_task.keys())

    def report(self, seed, success):
        if self.test_mode and seed in self.total_per_task:
            return

        self.total_per_task[seed] += 1
        if success:
            self.success_per_task[seed] += 1

        self.success_pc_per_task[seed] = self.success_per_task[seed] / self.total_per_task[seed]

    def get_task_from_seed(self, seed, agent):
        np.random.seed(seed)

        quantifier = ColorNumberQuantifier(seed)
        mission = np.random.choice(list(mission_types.values())[:self.n_missions])

        task = Task(mission, quantifier, agent, args.world_size, args.cell_size, seed)

        return task

    @staticmethod
    def get_task_hash(mission, quantifier):
        return '%s_%s' % (mission.name, quantifier.get_hash())
