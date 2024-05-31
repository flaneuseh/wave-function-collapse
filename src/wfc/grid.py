import numpy as np

from wfc.cell import Cell
from wfc.pattern import Pattern
from wfc.propagator import Propagator


class Grid:
    """
    Grid is made of Cells
    """

    # TODO: propagate for all cells that change.
    def __init__(self, initial_state, pattern_size):
        self.pattern_size = pattern_size
        self.size = tuple(
            np.subtract(initial_state.shape, tuple(x - 1 for x in self.pattern_size))
        )
        self.grid = np.empty(self.size, dtype=object)
        for position in np.ndindex(self.size):
            cell = Cell(
                position,
                len(Pattern.index_to_pattern),
                self,
            )
            self.grid[position] = cell

        for position in np.ndindex(self.size):
            cell = self.grid[position]
            old_allowed = set(cell.allowed_patterns)
            new_allowed = old_allowed & set(
                Pattern.filter_on(
                    Pattern.partial_at(initial_state, pattern_size, position)
                )
            )
            if new_allowed != old_allowed:
                cell.allowed_patterns = list(new_allowed)
                Propagator.propagate(cell)

    def find_lowest_entropy(self):
        min_entropy = 999999
        lowest_entropy_cells = []
        for cell in self.grid.flat:
            if cell.is_stable():
                continue

            entropy = cell.entropy()

            if entropy == min_entropy:
                lowest_entropy_cells.append(cell)
            elif entropy < min_entropy:
                min_entropy = entropy
                lowest_entropy_cells = [cell]

        if len(lowest_entropy_cells) == 0:
            return None
        cell = lowest_entropy_cells[np.random.randint(len(lowest_entropy_cells))]
        return cell

    def get_cell(self, index):
        """
        Returns the cell contained in the grid at the provided index
        :param index: (...z, y, x)
        :return: cell
        """
        return self.grid[index]

    def get_grid(self):
        return self.grid

    def get_image(self):
        """
        Returns the grid converted from index to back to color
        :return:
        """
        image = np.zeros(
            tuple(np.add(self.grid.shape, tuple(x - 1 for x in self.pattern_size)))
        )
        for index in np.ndindex(image.shape):
            grid_index = list(index)
            offset = [0, 0, 0]
            for i, d in enumerate(index):
                max_d = self.grid.shape[i] - 1
                if d > max_d:
                    grid_index[i] = max_d
                    offset[i] = d - max_d

            cell = self.grid[tuple(grid_index)]
            image[index] = cell.get_value(tuple(offset))
        image = Pattern.index_to_img(image)
        return image

    def check_contradiction(self):
        for cell in self.grid.flat:
            if len(cell.allowed_patterns) == 0:
                return True
        return False

    def print_allowed_pattern_count(self):
        grid_allowed_patterns = np.vectorize(lambda c: len(c.allowed_patterns))(
            self.grid
        )
        print(grid_allowed_patterns)
