import numpy as np

from config import args
from environment.objects.grid_object import Object


class Collectable(Object):
    def __init__(self, color, shape=None, full_mask=False, colorized=False, dropable=False):
        super().__init__()
        self.color = color
        self.shape = shape
        self.collected = False
        self.colorized = colorized
        self.full_mask = full_mask
        self.dropable = dropable

    def render(self):
        if self.colorized or self.dropable:
            border_color = (255,105,180) if self.colorized else (50,50,50)
            r = np.ones((args.cell_size, args.cell_size, 3)) * border_color
            r[1, 1] = self.color
            self.full_mask = True
        else:
            if self.shape is None:
                r = np.ones((args.cell_size, args.cell_size, 3)) * self.color
            else:
                r = np.ones((args.cell_size, args.cell_size, 3)) * (255, 255, 255)
                self.full_mask = True
                if self.shape == 0:
                    r[1, 1] = self.color
                elif self.shape == 1:
                    r[1, :] = self.color
                    r[:, 1] = self.color
                elif self.shape == 2:
                    r[:, :] = self.color
                elif self.shape == 3:
                    r[0, :] = self.color
                    r[2, :] = self.color
                elif self.shape == 4:
                    r[:, 0] = self.color
                    r[:, 2] = self.color

        return r

    def mask(self):
        if self.full_mask:
            mask = np.ones((args.cell_size, args.cell_size))
        else:
            mask = np.zeros((args.cell_size, args.cell_size))
            mask[1,1] = 1
        return mask
