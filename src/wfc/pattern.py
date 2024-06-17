import numpy as np
import copy
from wfc.utils import islistlike


class Pattern:
    """
    Pattern is a configuration of tiles from the input image.
    """

    # Class variables
    index_to_pattern = {}
    color_to_index = {(-1, -1, -1): -1, "": -1}
    index_to_color = {}
    patterns = []
    pattern_index = 0

    # Data format. Allowable values:
    # rbg: standard colour image
    # char: character representation
    format = "rbg"

    # Data transforms on the grid
    # local - transforms applied to each tile
    local_transforms = {
        "flipx": False,  # left-right (axis 2) flip
        "flipy": False,  # up-down (axis 1) flip
        "fliph": False,  # height (axis 0) flip
        "rotxy": [],  # rotate clockwise around the xy (axes 2, 1) plane by 90 degrees times numbers specified by list
        "rotxh": [],  # rotate clockwise around the yh (axes 1, 0) plane by 90 degrees times numbers specified by list
        "rotyh": [],  # rotate clockwise around the xh (axes 2, 0) plane by 90 degrees times numbers specified by list
    }
    # global - transforms applied to the entire sample
    global_transforms = {
        "flipx": False,  # left-right (axis 2) flip
        "flipy": False,  # up-down (axis 1) flip
        "fliph": False,  # height (axis 0) flip
        "rotxy": [],  # rotate clockwise around the xy (axes 2, 1) plane by 90 degrees times numbers specified by list
        "rotxh": [],  # rotate clockwise around the yh (axes 1, 0) plane by 90 degrees times numbers specified by list
        "rotyh": [],  # rotate clockwise around the xh (axes 2, 0) plane by 90 degrees times numbers specified by list
    }

    # Where not specified, use numpy padding settings (see https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
    padding = {
        # "axis_order": (2, 1, 0),  # The order with which to pad the 3 axes, which numpy does not provide controls for. Defaults to the numpy default, which pads in ascending order.
        # "constant_values": () # Constants to pad with in each of the 3 axes. Should be the same datatype as the input array, and should match the shape of the 4th axis and above.
    }

    @staticmethod
    def set_format(format):
        Pattern.format = format

    @staticmethod
    def set_local_transforms(transforms):
        for key, value in transforms.items():
            Pattern.local_transforms[key] = value

    @staticmethod
    def set_global_transforms(transforms):
        for key, value in transforms.items():
            Pattern.global_transforms[key] = value

    @staticmethod
    def set_padding(transforms):
        for key, value in transforms.items():
            Pattern.padding[key] = value

    def __init__(self, data, index):
        self.index = index
        self.data = np.array(data)
        self.legal_patterns_index = {}  # offset -> [pattern_index]

    def get(self, index=None):
        if index is None:
            return self.data.item(0)
        return self.data[index]

    def set_legal_patterns(self, offset, legal_patterns):
        self.legal_patterns_index[offset] = legal_patterns

    @property
    def shape(self):
        return self.data.shape

    def is_compatible(self, candidate_pattern, offset):
        """
        Check if pattern is compatible with a candidate pattern for a given offset
        :param candidate_pattern:
        :param offset:
        :return: True if compatible
        """
        assert self.shape == candidate_pattern.shape

        # Precomputed compatibility
        if offset in self.legal_patterns_index:
            return candidate_pattern.index in self.legal_patterns_index[offset]

        # Computing compatibility
        ok_constraint = True
        start = tuple([max(offset[i], 0) for i, _ in enumerate(offset)])
        end = tuple([
            min(self.shape[i] + offset[i], self.shape[i]) for i, _ in enumerate(offset)
        ])
        for index in np.ndindex(end):  # index = (x, y, z...)
            start_constraint = True
            for i, d in enumerate(index):
                if d < start[i]:
                    start_constraint = False
                    break
            if not start_constraint:
                continue

            if candidate_pattern.get(
                tuple(np.array(index) - np.array(offset))
            ) != self.get(index):
                ok_constraint = False
                break

        return ok_constraint

    def to_image(self):
        return Pattern.index_to_img(self.data)

    @staticmethod
    def filter_on(partial):
        valid_indices = []
        for pi, pattern in Pattern.index_to_pattern.items():
            is_valid = True
            for i in np.ndindex(partial.shape):
                if partial[i] != -1 and partial[i] != pattern.get(i):
                    is_valid = False
                    break
            if is_valid:
                valid_indices.append(pi)
        return valid_indices

    @staticmethod
    def partial_at(grid, pattern_size, position):
        shape = grid.shape
        out = False
        for i, d in enumerate(position):  # d is a dimension, e.g.: x, y, z
            if d > shape[i] - pattern_size[i]:
                out = True
                break
        if out:
            return np.full(pattern_size, fill_value=-1, dtype=int)

        pattern_location = [
            range(d, pattern_size[i] + d) for i, d in enumerate(position)
        ]
        pattern_data = grid[np.ix_(*pattern_location)]
        return pattern_data

    @staticmethod
    def from_samples(samples, pattern_size):
        for sample in samples:
            transforms = [sample]
            shape = sample.shape
            if Pattern.global_transforms["flipx"]:
                transforms.append(np.flip(sample, axis=2))
            if shape[1] > 1:  # is 2D
                if Pattern.global_transforms["flipy"]:
                    transforms.append(np.flip(sample, axis=1))
                for n in Pattern.global_transforms["rotxy"]:
                    transforms.append(np.rot90(sample, n, axes=(2, 1)))

            if shape[0] > 1:  # is 3D
                if Pattern.global_transforms["fliph"]:
                    transforms.append(np.flip(sample, axis=0))
                for n in Pattern.global_transforms["rotxh"]:
                    transforms.append(np.rot90(sample, n, axes=(2, 0)))
                for n in Pattern.global_transforms["rotyh"]:
                    transforms.append(np.rot90(sample, n, axes=(1, 0)))

            for transform in transforms:
                Pattern.from_sample(transform, pattern_size)
        return Pattern.patterns

    @staticmethod
    def from_sample(sample, pattern_size):
        """
        Compute patterns from sample
        :param pattern_size:
        :param sample:
        :return: list of patterns
        """

        sample = Pattern.img_to_indexes(sample)
        sample = Pattern.pad(sample)

        shape = sample.shape

        for index, _ in np.ndenumerate(sample):
            # Checking if index is out of bounds
            out = False
            for i, d in enumerate(index):  # d is a dimension, e.g.: x, y, z
                if d > shape[i] - pattern_size[i]:
                    out = True
                    break
            if out:
                continue

            pattern_location = [
                range(d, pattern_size[i] + d) for i, d in enumerate(index)
            ]
            pattern_data = sample[np.ix_(*pattern_location)]
            datas = [pattern_data]
            if Pattern.local_transforms["flipx"]:
                datas.append(np.flip(pattern_data, axis=2))
            if shape[1] > 1:  # is 2D
                if Pattern.local_transforms["flipy"]:
                    datas.append(np.flip(pattern_data, axis=1))
                for n in Pattern.local_transforms["rotxy"]:
                    datas.append(np.rot90(pattern_data, n, axes=(2, 1)))

            if shape[0] > 1:  # is 3D
                if Pattern.local_transforms["fliph"]:
                    datas.append(np.flip(pattern_data, axis=0))
                for n in Pattern.local_transforms["rotxh"]:
                    datas.append(np.rot90(pattern_data, n, axes=(2, 0)))
                for n in Pattern.local_transforms["rotyh"]:
                    datas.append(np.rot90(pattern_data, n, axes=(1, 0)))

            # Checking existence
            # TODO: more probability to multiple occurrences when observe phase
            for data in datas:
                exist = False
                for p in Pattern.patterns:
                    if (p.data == data).all():
                        exist = True
                        break
                if exist:
                    continue

                pattern = Pattern(data, Pattern.pattern_index)
                Pattern.patterns.append(pattern)
                Pattern.index_to_pattern[Pattern.pattern_index] = pattern
                Pattern.pattern_index += 1

        return Pattern.patterns

    @staticmethod
    def img_to_indexes(sample):
        """
        Convert a rgb image to a 2D array with pixel index
        :param sample:
        :return: pixel index sample
        """
        indexes = []

        idx_shape = sample.shape
        if Pattern.format == "rgb":
            idx_shape = idx_shape[:-1]  # without last rgb dim

        indexes = np.full(idx_shape, fill_value=-1, dtype=int)
        val_idx = len(Pattern.index_to_color)
        for index in np.ndindex(idx_shape):
            val = sample[index]
            if Pattern.format == "rgb":
                val = tuple(val)
            if val not in Pattern.color_to_index:
                Pattern.color_to_index[val] = val_idx
                Pattern.index_to_color[val_idx] = val
                val_idx += 1

            indexes[index] = Pattern.color_to_index[val]

        return indexes

    def pad(grid):
        settings = Pattern.padding
        if "pad_width" not in settings or len(settings["pad_width"]) == 0:
            return grid

        args = copy.deepcopy(settings)
        pad_width = list(args["pad_width"])
        while len(pad_width) < len(grid.shape):
            pad_width.append((0, 0))
        args["pad_width"] = tuple(pad_width)

        constant_values = args["constant_values"]
        if not islistlike(constant_values):
            constant_values = [constant_values]

        constant_vals_as_indexes = []
        for dim_vals in constant_values:
            if not islistlike(dim_vals):
                dim_vals = [dim_vals]
            dim_vals_as_indexes = []
            for end in dim_vals:
                if islistlike(end):
                    end = tuple(end)
                if end not in Pattern.color_to_index:
                    next_idx = len(Pattern.index_to_color)
                    Pattern.index_to_color[next_idx] = end
                    Pattern.color_to_index[end] = next_idx
                end = Pattern.color_to_index[end]
                dim_vals_as_indexes.append(end)
            if len(dim_vals_as_indexes) == 1:
                dim_vals_as_indexes = dim_vals_as_indexes[0]
            constant_vals_as_indexes.append(dim_vals_as_indexes)
        if len(constant_vals_as_indexes) == 1:
            constant_vals_as_indexes = constant_vals_as_indexes[0]
        else:
            while len(constant_vals_as_indexes) < len(grid.shape):
                constant_vals_as_indexes.append(constant_vals_as_indexes[0])
            constant_vals_as_indexes = tuple(constant_vals_as_indexes)

        args["constant_values"] = constant_vals_as_indexes

        if "axis_order" in args:
            del args["axis_order"]
        if "axis_order" not in settings or len(settings["axis_order"]) == 0:
            return np.pad(grid, **args)

        L = grid
        for axis in settings["axis_order"]:
            set_pad_width = np.zeros((len(L.shape), 2), dtype=int)
            set_pad_width[axis] = settings["pad_width"][axis]
            args["pad_width"] = tuple(set_pad_width)
            L = np.pad(L, **args)

        return L

    @staticmethod
    def index_to_img(sample):
        color = next(iter(Pattern.index_to_color.values()))

        type = float
        if Pattern.format == "char":
            type = str
        shape = sample.shape
        if Pattern.format == "rbg":
            shape += len(color)
        image = np.zeros(shape, dtype=type)
        for index in np.ndindex(sample.shape):
            pattern_index = sample[index]
            if pattern_index == -1:
                if Pattern.format == "rbg":
                    image[index] = [0.5 for _ in range(len(color))]  # Grey
            else:
                image[index] = Pattern.index_to_color[pattern_index]
        return image

    @staticmethod
    def from_index(pattern_index):
        return Pattern.index_to_pattern[pattern_index]
