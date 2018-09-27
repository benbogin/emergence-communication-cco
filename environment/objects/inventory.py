import numpy as np

from utils.render import draw_square


class Inventory:
    """
     Inventory implemented as a queue (FIFO),
      objects are added during goal construction or during play
    """
    def __init__(self, size, pixels_per_cell):
        self.pixels_per_cell = pixels_per_cell
        self.max_size = size    # maximum size of the inventory
        self.size = 0           # current size of the inventory
        self.objects_queue = []
        self.is_valid = True    # flag to signal when the agent performed an action that should fail him

    def reset(self):
        self.objects_queue = []
        self.size = 0
        self.is_valid = True

    def add_object(self, object):
        if not self.is_full():
            self.objects_queue.insert(0, object)
            self.size += 1
            object.inventory = self
        else:
            # fail task - another option is to simply ignore
            self.is_valid = False

    def remove_object(self):
        assert not self.is_empty()
        obj = self.objects_queue[0]
        self.objects_queue = self.objects_queue[1:]
        self.size -= 1
        return obj

    def get_pos(self, object):
        return self.objects_queue.index(object)

    def render(self, scale=1, borders=False):
        ppc = self.pixels_per_cell * scale
        if borders:
            ppc += 1
        window_size = self.max_size * ppc
        screen = np.ones((ppc, window_size, 3), dtype=np.uint8) * 255

        for index, obj in enumerate(self.objects_queue):
            object_rendered = np.repeat(np.repeat(obj.render(), scale, axis=0), scale, axis=1)
            mask = np.expand_dims(np.repeat(np.repeat(obj.mask(), scale, axis=0), scale, axis=1), axis=2)

            screen_pos = index
            x = 0
            x2 = ppc - (1 if borders else 0)
            y = screen_pos * ppc
            y2 = screen_pos * ppc + ppc - (1 if borders else 0)
            screen[x:x2, y:y2, :] = screen[x:x2, y:y2, :]*(1-mask) + object_rendered*mask

        # if borders:
        #     for i in range(1, self.max_size):
        #         draw_square(screen, (0, i*ppc - 1), (self.max_size*ppc, i*ppc), 0)
        #         draw_square(screen, (i*ppc - 1, 0), (i*ppc, self.max_size*ppc), 0)

        return screen

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.max_size
