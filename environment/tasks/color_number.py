from collections import Counter
import random
from config import args

from environment.objects.collectable import Collectable


class ColorNumberQuantifier:
    def __init__(self, seed):
        self.random = random.Random(seed)

        self.all_colors = [((255, 0, 0), "RED"), ((0, 0, 255), "BLUE"), ((255, 225, 25), "YELLOW"),
                           ((128, 0, 0), "MAROON"), ((128, 128, 128), "GRAY"), ((0, 255, 255), "CYAN"),
                           ((128, 0, 128), "PURPLE"), ((128, 128, 0), "OLIVE"), ((0, 0, 0), "BLACK"),
                           ((255,0,255), "MAGENTA"), ((0,128,128), "TEAL"), ((0,0,128), "NAVY")]
        self.rel_colors = self.all_colors[:args.n_colors]

        self.color_to_string = dict(self.rel_colors)

        self.colors = list(self.color_to_string.keys())
        self.random.shuffle(self.colors)

        self.objective_dict = Counter()

    def get_objects(self):
        objects = []
        inventory_objects = []

        max_collect = 3

        objective_color = self.colors[0]
        distraction_color = self.colors[1]
        objective_number = self.random.randint(1, max_collect)

        for i in range(max_collect):
            objects.append(Collectable(color=objective_color))
            objects.append(Collectable(color=distraction_color))

        for i in range(objective_number):
            inventory_objects.append(Collectable(color=objective_color))

        self.objective_dict[(objective_color, None)] = objective_number
        return objects, inventory_objects

    def is_success(self, acted_objects):
        acted_dict = self.create_acted_dict_from_list(acted_objects)
        attributes = set(acted_dict.keys()) | set(self.objective_dict.keys())

        for attribute in attributes:
            if acted_dict[attribute] != self.objective_dict[attribute]:
                return False

        return True

    def is_fail(self, acted_objects):
        acted_dict = self.create_acted_dict_from_list(acted_objects)
        attributes = set(acted_dict.keys()) | set(self.objective_dict.keys())

        for attribute in attributes:
            if acted_dict[attribute] > self.objective_dict[attribute]:
                return True

        return False

    @staticmethod
    def create_acted_dict_from_list(acted_objects):
        acted_dict = Counter()
        for obj in acted_objects:
            acted_dict[obj.color, obj.shape] += 1
        return acted_dict

    def get_hash(self):
        return '_'.join([(self.color_to_string[color] + '_' + str(value)) for (color, shape), value in self.objective_dict.items()])
