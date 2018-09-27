import numpy as np

from environment.objects.landmark import Landmark
from environment.tasks.missions import Missions
from environment.objects.world import World
from environment.objects.inventory import Inventory


class Task:
    def __init__(self, mission, quantifier, agent, world_size, cell_size, seed=None):
        self.agent = agent
        self.cell_size = cell_size
        self.world_size = world_size
        self.world = None
        self.agent_location = None
        self.inventory = None
        self.agent.inventory = None

        self.mission = mission
        self.quantifier = quantifier

        self.seed = seed

        self.acted_objects = []
        self.declared_finished = False
        self.acted_wrong_mission = False
        self.deserves_bonus = False

        self.random_horizontal_flip = True
        self.horizontal_flipped = False

        self.dropzone_locations = [(0,0), (0,1), (0, 2), (0,3), (0,4)]

    def initialize_world_with_new_task(self):
        np.random.seed(self.seed)
        self.world = World(self.world_size, pixels_per_cell=self.cell_size, current_task=self)
        self.inventory = Inventory(self.world_size, pixels_per_cell=self.cell_size)
        self.agent.inventory = Inventory(self.world_size, pixels_per_cell=self.cell_size)

        self.place_players()
        self.place_objects()

        self.horizontal_flipped = bool(np.random.randint(2)) and self.random_horizontal_flip

        return self.world, self.inventory

    def add_to_world(self, object, location=None, exclude_pos=None, world=None):
        if world is None:
            world = self.world
        if location is None:
            location = self.get_random_location(exclude_pos)

        world.add_object(object, location)
        return location

    def get_random_location(self, exclude_pos):
        location = (np.random.randint(self.world_size), np.random.randint(self.world_size))
        while exclude_pos is not None and location in exclude_pos:
            location = (np.random.randint(self.world_size), np.random.randint(self.world_size))
        return location

    def place_players(self):
        self.agent_location = self.add_to_world(self.agent, location=(2,2))

    def get_player_pos(self):
        return self.agent.get_pos()

    def is_fail(self):
        if self.acted_wrong_mission or self.quantifier.is_fail(self.acted_objects) or self.quantifier.is_fail(self.agent.inventory.objects_queue):
            return True

        return self.declared_finished and not self.quantifier.is_success(self.acted_objects)

    def is_success(self):
        if self.acted_wrong_mission:
            return False

        success = self.quantifier.is_success(self.acted_objects)
        return success and self.declared_finished

    def get_bonus(self):
        # agent receives bonus even if failed - this helps learning
        if self.deserves_bonus:
            self.deserves_bonus = False
            return 0.2
        else:
            return 0

    def place_objects(self):
        for loc in self.dropzone_locations:
            self.add_to_world(Landmark(color=(50,50,50), collectable_target=True), location=loc, world=self.world)

        objects, inventory_objects = self.quantifier.get_objects()
        exclude_locations = [self.agent.get_pos()] + self.dropzone_locations

        for i, obj in enumerate(objects):
            loc = self.get_random_location(exclude_pos=exclude_locations)
            self.world.add_object(obj, loc)
            exclude_locations.append(loc)

        for obj in inventory_objects:
            if self.mission == Missions.Colorize:
                obj.colorized = True
            if self.mission == Missions.Bring:
                obj.dropable = True
            self.inventory.add_object(obj)

    def object_collected(self, obj):
        if self.mission == Missions.Collect:
            self.acted_objects.append(obj)

        if self.mission != Missions.Colorize:
            self.deserves_bonus = True

    def object_colorized(self, obj):
        if self.mission == Missions.Colorize:
            self.deserves_bonus = True
            self.acted_objects.append(obj)
        else:
            self.acted_wrong_mission = True

    def objects_dropped(self):
        self.deserves_bonus = True
        if self.mission == Missions.Bring:
            self.acted_objects = self.agent.inventory.objects_queue
            self.declared_finished = True
        else:
            self.acted_wrong_mission = True

    def declare_finished(self):
        if len(self.agent.inventory.objects_queue) > 0 and self.mission != Missions.Bring:
            self.declared_finished = True

    def agent_moves_noncolorized_object(self):
        if self.mission == Missions.Colorize:
            # if mission is Colorize and agent moved before colorizing inventory, task failed
            self.acted_wrong_mission = True

    def get_hash(self):
        return '%s_%s' % (self.mission.name, self.quantifier.get_hash())
