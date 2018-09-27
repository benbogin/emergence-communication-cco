import numpy as np

from config import args
from environment.objects.grid_object import Object


class Landmark(Object):
    def __init__(self, color, collectable_target=False):
        super().__init__()
        self.color = color
        self.collectable_target = collectable_target
        self.used = False

    def render(self):
        pixels = np.ones((args.cell_size, args.cell_size, 3)) * self.color

        return pixels

    def mask(self):
        mask = np.ones((args.cell_size, args.cell_size))
        mask[1,1] = 0
        return mask

    def is_dropable(self):
        return self.collectable_target and not self.used
