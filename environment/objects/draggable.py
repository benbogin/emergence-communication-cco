import numpy as np

from config import args
from environment.objects.grid_object import Object


class Draggable(Object):
    def __init__(self, color):
        super().__init__()
        self.color = color

    def render(self):
        return np.ones((args.cell_size, args.cell_size, 3)) * self.color

    def mask(self):
        mask = np.zeros((args.cell_size, args.cell_size))
        mask[1,1] = 1
        return mask
