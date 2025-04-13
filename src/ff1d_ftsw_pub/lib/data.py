from __future__ import annotations

import abc
import enum
import importlib
import json
import pathlib
import tomllib
from typing import Any, Self

import attrs
import numpy as np

_manifest_name = "manifest.toml"


def read_json(handle: pathlib.Path) -> dict:
    with open(handle) as f:
        values = json.load(f)
    return values


def read_csv(handle: pathlib.Path):
    return np.loadtxt(handle, delimiter=",")


def read_data(subdir: pathlib.Path) -> dict[str, Any]:
    with open(subdir.joinpath(_manifest_name), "rb") as file:
        manifest = tomllib.load(file)

    return {
        pathlib.Path(handle).stem: getattr(FileFormat, file_format)(
            subdir.joinpath(handle)
        )
        for file_format in manifest
        for handle in manifest[file_format]["files"]
    }


class FileFormat(enum.Enum):
    csv = read_csv
    json = read_json
    npz = np.load

    def __call__(self, handle: pathlib.Path):
        return self.value(handle)


class Loader(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_raw_data(cls, raw_data: dict): ...

    @classmethod
    def from_label(cls, label) -> Loader:
        subdir = importlib.resources.files("ff1d_ftsw_pub.data").joinpath(label)
        with importlib.resources.as_file(subdir) as physical_subdir:
            if label == "simple_example":
                return SimpleExampleLoader.from_raw_data(read_data(physical_subdir))
            else:
                raise NotImplementedError


@attrs.frozen
class SimpleExampleLoader(Loader):
    nondim: np.ndarray
    jumps: np.ndarray
    variables: dict

    @classmethod
    def from_raw_data(cls, raw_data) -> Self:
        nondim, jumps, variables = cls._extract(raw_data)
        variables = cls._clean(variables)
        return cls(nondim, jumps, variables)

    @staticmethod
    def _extract(raw_data):
        nondim = (
            raw_data["parameters"]["varnish"]["flexural_length"]
            * raw_data["results"]["wavenumbers"]
        )
        variables = {
            k: raw_data["results"][k]
            for k in (
                "relaxation_lengths",
                "amplitude_thresholds",
                "curvature_thresholds",
                "normalised_fractures",
            )
        }
        return nondim, raw_data["jumps"], variables

    @staticmethod
    def _clean_fracture_locations(fracture_locations: np.ndarray) -> np.ndarray:
        _m = fracture_locations > 0.5
        fracture_locations[_m] = 1 - fracture_locations[_m]
        return fracture_locations

    @staticmethod
    def _clean_curvature(critical_curvatures: np.ndarray) -> np.ndarray:
        return np.abs(critical_curvatures)

    @staticmethod
    def _clean(variables: dict) -> dict:
        variables["normalised_fractures"] = (
            SimpleExampleLoader._clean_fracture_locations(
                variables["normalised_fractures"]
            )
        )
        variables["curvature_thresholds"] = SimpleExampleLoader._clean_curvature(
            variables["curvature_thresholds"]
        )
        return variables


@attrs.frozen
class JointDensityLoader(Loader):
    thicknesses: np.ndarray
    youngs_moduli: np.ndarray

    @classmethod
    def from_raw_data(cls, raw_data) -> Self:
        _h, _Y = (raw_data[k] for k in ("thicknesses", "youngs_moduli"))
        return cls(_h, _Y)
