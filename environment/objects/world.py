from collections import OrderedDict, defaultdict

import numpy as np

from environment.action import Action
from environment.objects.agent import Agent
from environment.objects.collectable import Collectable
from environment.objects.draggable import Draggable
from environment.objects.landmark import Landmark
from utils.render import draw_square


class World:
    def __init__(self, size, pixels_per_cell, current_task):
        self.pixels_per_cell = pixels_per_cell
        self.size = size
        self.dragging = None
        self.current_task = current_task
        self.reset()
        self.horizontal_flipped = False

    def reset(self):
        self.objects_locations = OrderedDict()
        self.locations_to_object = defaultdict(list)
        self.dragging = None

    def add_object(self, object, pos):
        self.objects_locations[object] = pos
        self.locations_to_object[pos].append(object)
        object.world = self

    def remove_object(self, obj):
        pos = self.objects_locations[obj]
        del self.objects_locations[obj]
        self.locations_to_object[pos].remove(obj)

    def agent_act(self, agent, action):
        if action == Action.UP:
            self.move_agent(agent, (-1, 0))
        if action == Action.RIGHT:
            self.move_agent(agent, (0, 1))
        if action == Action.DOWN:
            self.move_agent(agent, (1, 0))
        if action == Action.LEFT:
            self.move_agent(agent, (0, -1))
        if action == Action.USE:
            self.use_object(agent)
        if action == Action.DONE:
            self.current_task.declare_finished()

    def move_agent(self, agent, direction):
        curr_pos = self.objects_locations[agent]
        if curr_pos[0] + direction[0] < 0 or curr_pos[0] + direction[0] >= self.size:
            return False

        if curr_pos[1] + direction[1] < 0 or curr_pos[1] + direction[1] >= self.size:
            return False

        self.objects_locations[agent] = new_pos = (curr_pos[0] + direction[0], curr_pos[1] + direction[1])
        self.locations_to_object[curr_pos].remove(agent)
        self.locations_to_object[new_pos].append(agent)

        if not agent.inventory.is_empty() and not agent.inventory.objects_queue[0].colorized:
            self.current_task.agent_moves_noncolorized_object()

        if self.dragging:
            self.objects_locations[self.dragging] = new_pos
        else:
            for obj, pos in self.objects_locations.items():
                if pos == new_pos and type(obj) is Draggable:
                    self.dragging = obj
                    break

        obj = self.get_object_in_pos(agent.get_pos())
        if obj is not None and type(obj) is Collectable:
            self.collect_object(agent, obj)

        if obj is not None and type(obj) is Landmark and obj.is_dropable() and not agent.inventory.is_empty():
            self.drop_collected_object(agent, obj)

        return True

    def get_object_in_pos(self, pos):
        objs_in_pos = [x for x in self.locations_to_object[pos] if type(x) != Agent]
        assert(len(objs_in_pos) <= 1)
        return objs_in_pos[0] if len(objs_in_pos) == 1 else None

    def collect_object(self, agent, obj):
        agent.inventory.add_object(obj)
        obj.collected = True
        self.remove_object(obj)
        self.current_task.object_collected(obj)

    def colorize_object(self, agent, obj):
        if not obj.colorized:
            obj.colorized = True
            self.current_task.object_colorized(obj)

    def drop_collected_object(self, agent, obj):
        self.current_task.objects_dropped()

    def use_object(self, agent):
        obj = self.get_object_in_pos(agent.get_pos())

        # we should make sure that the agent doesn't
        #  need to 'use' more than max_size objects
        if obj is not None:
            if type(obj) is Collectable and not obj.colorized:
                self.collect_object(agent, obj)
        else:
            if len(agent.inventory.objects_queue) > 0 and type(agent.inventory.objects_queue[0]) is Collectable:
                self.colorize_object(agent, agent.inventory.objects_queue[0])
        # trying to pick an object from an empty position
        # else:
        #     agent.inventory.is_valid = False

    def get_pos(self, object, flip_horizontal=False):
        if not flip_horizontal:
            return self.objects_locations[object]
        else:
            return self.objects_locations[object][0], self.size - 1 - self.objects_locations[object][1]

    def render(self, pov=None, scale=1, borders=False, object_to_hide=None, flip_horizontal=False):
        ppc = self.pixels_per_cell * scale
        if borders:
            ppc += 1
        window_size = self.size * ppc
        screen = np.ones((window_size, window_size, 3), dtype=np.uint8) * 255

        center_cell = (self.size // 2, self.size // 2)

        objects = list(self.objects_locations.keys())
        sorted(objects, key=lambda o: 2 if o == pov else 1 if type(o) is Draggable else 0)

        # pov = None

        for index, obj in enumerate(objects):
            if obj == object_to_hide:
                continue

            object_rendered = np.repeat(np.repeat(obj.render(), scale, axis=0), scale, axis=1)
            mask = np.expand_dims(np.repeat(np.repeat(obj.mask(), scale, axis=0), scale, axis=1), axis=2)

            abs_pos = self.get_pos(obj, flip_horizontal=flip_horizontal and type(obj) is not Agent)
            if pov is not None:
                rel_pos = np.subtract(abs_pos, self.get_pos(pov))
                screen_pos = np.add(center_cell, rel_pos)
            else:
                screen_pos = abs_pos

            if screen_pos[0] < 0 or screen_pos[1] < 0 or screen_pos[0] >= self.size or screen_pos[1] >= self.size:
                continue

            x = screen_pos[0] * ppc
            x2 = screen_pos[0] * ppc + ppc - (1 if borders else 0)
            y = screen_pos[1] * ppc
            y2 = screen_pos[1] * ppc + ppc - (1 if borders else 0)
            screen[x:x2, y:y2, :] = screen[x:x2, y:y2, :]*(1-mask) + object_rendered*mask

        # render black where out of world
        if pov is not None:
            for i in range(self.size):
                for j in range(self.size):
                    rel_pos = (i, j)
                    abs_pos = np.subtract(pov.get_pos(), center_cell)
                    abs_pos = np.add(abs_pos, (i,j))
                    if abs_pos[0] < 0 or abs_pos[1] < 0 or abs_pos[0] >= self.size or abs_pos[1] >= self.size:
                        x = rel_pos[0] * ppc
                        x2 = rel_pos[0] * ppc + ppc
                        y = rel_pos[1] * ppc
                        y2 = rel_pos[1] * ppc + ppc
                        screen[x:x2, y:y2, :] = (0, 0, 0)

        if borders:
            for i in range(1, self.size):
                draw_square(screen, (0, i*ppc - 1), (self.size*ppc, i*ppc), 0)
                draw_square(screen, (i*ppc - 1, 0), (i*ppc, self.size*ppc), 0)

        return screen
