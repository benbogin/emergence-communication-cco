import numpy as np

from config import args
from environment.objects.grid_object import Object


class Agent(Object):
    def __init__(self, color, current_task=None):
        super().__init__()
        self.current_task = current_task
        self.color = color

    def render(self):
        return np.ones((args.cell_size, args.cell_size, 3)) * self.color

    def equals(self, other):
        return self.color == other.color
