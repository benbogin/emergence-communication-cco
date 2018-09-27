from config import args
import numpy as np


class Object:
    def __init__(self):
        self.world = None
        self.inventory = None

    def get_pos(self):
        return self.world.get_pos(self)

    def mask(self):
        return np.ones((args.cell_size, args.cell_size))