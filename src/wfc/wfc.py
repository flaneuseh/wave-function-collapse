import time
import numpy as np
import copy

from wfc.grid import Grid
from wfc.pattern import Pattern
from wfc.propagator import Propagator

"""
Implementation of WaveFunctionCollapse
Following the "WaveFunctionCollapse is Constraint Solving in the Wild" terminology
"""


class WaveFunctionCollapse:
    """
    WaveFunctionCollapse encapsulates the wfc algorithm
    """

    @staticmethod
    def unset_padding(transforms):
        for key in transforms:
            if key in WaveFunctionCollapse.padding:
                del WaveFunctionCollapse.padding[key]

    @staticmethod
    def default_initialize_fn(grid_size):
        return np.full(grid_size, fill_value=-1, dtype=int)

    def __init__(
        self, grid_size, samples, pattern_size, initialize_fn=default_initialize_fn
    ):
        self.grid_size = grid_size
        self.pattern_size = pattern_size
        self.patterns = Pattern.from_samples(samples, pattern_size)
        self.propagator = Propagator(self.patterns)
        self.initialize_fn = initialize_fn

    def run(self, debug=False):
        start_time = time.time()

        self.grid = self._create_grid(self.grid_size)
        done = False
        while not done:
            done = self.step(debug)

        print("WFC run took %s seconds" % (time.time() - start_time))

    def step(self, debug=False):
        if debug:
            self.grid.print_allowed_pattern_count()
        cell = self.observe()
        if cell is None:
            return True
        self.propagate(cell)
        return False

    def get_initial_image(self):
        return self.initial_image

    def get_image(self):
        return self.grid.get_image()

    def get_patterns(self):
        return [pattern.to_image() for pattern in self.patterns]

    def observe(self):
        if self.grid.check_contradiction():
            return None
        cell = self.grid.find_lowest_entropy()

        if cell is None:
            return None

        cell.choose_rnd_pattern()

        return cell

    def propagate(self, cell):
        Propagator.propagate(cell)

    def _create_grid(self, grid_size):
        self.initial_image = self.initialize_fn(grid_size)
        initial_state = Pattern.pad(Pattern.img_to_indexes(self.initial_image))
        grid = Grid(initial_state, self.pattern_size)
        return grid
