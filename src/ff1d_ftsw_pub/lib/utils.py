import enum
import json

import numpy


def read_json(handle: pathlib.Path) -> dict:
    with open(handle) as f:
        values = json.load(f)
    return values


def read_csv(handle: pathlib.Path):
    return np.loadtxt(handle, delimiter=",")


class FileFormat(enum.Enum):
    csv = read_csv
    json = read_json
    npz = np.load

    def __call__(self, handle: pathlib.Path):
        return self.value(handle)


class FigureMatcher(enum.Enum):
    SCHEMATICS = 1
    FRACTURE_SEARCH = 2
    STRAIN_FRACTURE = 3
    SIMPLE_EXAMPLE = 4
    QUAD_REGIONS = 5
    QUAD_REGIONS_DISP = 6
    JOINT_DENSITY = 7
    ENS_AMPLITUDE = 8
    ENS_CONSTANTS = 9
