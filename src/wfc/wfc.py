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

    def __init__(self, grid_size, sample, pattern_size):
        self.pattern_size = pattern_size
        self.patterns = Pattern.from_sample(sample, pattern_size)
        self.propagator = Propagator(self.patterns)
        self.grid = self._create_grid(grid_size)

    def run(self, debug=False):
        start_time = time.time()

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
        self.propagator.propagate(cell)

    def _create_grid(self, grid_size):
        initial_state = Pattern.pad(np.full(grid_size, fill_value=-1, dtype=int))
        grid = Grid(initial_state, self.pattern_size)
        grid.print_allowed_pattern_count()
        for idx in np.ndindex(grid.get_grid().shape):
            cell = grid.get_cell(idx)
            if cell.is_stable():
                self.propagator.propagate(cell)
        return grid
