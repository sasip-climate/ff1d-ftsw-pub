from __future__ import annotations

import abc
import enum
import importlib
import json
import pathlib
import tomllib
from typing import Any, ClassVar, Self

import attrs
import numpy as np

from .utils import FigureMatcher

_manifest_name = "manifest.toml"

loaders_registry = dict()


def read_json(handle: pathlib.Path) -> dict:
    with open(handle) as f:
        return json.load(f)


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
    label: ClassVar[FigureMatcher]

    @classmethod
    def __attrs_init_subclass__(cls):
        loaders_registry[cls.label] = cls

    @classmethod
    @abc.abstractmethod
    def from_raw_data(cls, raw_data: dict): ...

    @classmethod
    def from_label(cls, label) -> Self:
        label_path = label.name.lower()
        subdir = importlib.resources.files("ff1d_ftsw_pub.data").joinpath(label_path)
        with importlib.resources.as_file(subdir) as physical_subdir:
            return loaders_registry[label].from_raw_data(read_data(physical_subdir))


def compute_freeboard(thickness: float, draught: float) -> float:
    return thickness - draught


@attrs.frozen
class SchematicsLoader(Loader):
    label: ClassVar[FigureMatcher] = FigureMatcher.SCHEMATICS
    draught: float
    freeboard: float
    variables: dict

    @classmethod
    def from_raw_data(cls, raw_data: dict) -> Self:
        draught, thickness, variables = cls._extract(raw_data)
        freeboard = compute_freeboard(thickness, draught)
        return cls(draught, freeboard, variables)

    @staticmethod
    def _extract(raw_data):
        draught, thickness = (
            raw_data["parameters"]["wuf"]["wui"]["ice"][key]
            for key in ("draft", "thickness")
        )
        return draught, thickness, raw_data["results"]


@attrs.frozen
class FractureSearchLoader(Loader):
    label: ClassVar[FigureMatcher] = FigureMatcher.FRACTURE_SEARCH
    draught: float
    freeboard: float
    energy_scalars: dict[str, float]
    energy_variables: dict[str, np.ndarray]
    deflection_variables: dict[str, np.ndarray]

    @classmethod
    def from_raw_data(cls, raw_data: dict) -> Self:
        draught, thickness, variables = cls._extract(raw_data)
        freeboard = compute_freeboard(thickness, draught)
        return cls(draught, freeboard, *variables)

    @staticmethod
    def _extract(raw_data):
        draught, thickness = (
            raw_data["parameters"]["wuf"]["wui"]["ice"][key]
            for key in ("draft", "thickness")
        )

        return (
            draught,
            thickness,
            (
                raw_data[k]
                for k in (
                    "fracture_values",
                    "energy_diagnostic",
                    "fracture_deflections",
                )
            ),
        )


@attrs.frozen
class StrainFractureLoader(Loader):
    label: ClassVar[FigureMatcher] = FigureMatcher.STRAIN_FRACTURE
    fracture_location: float
    threshold: float
    variables: dict[str, np.ndarray]

    @classmethod
    def from_raw_data(cls, raw_data: dict) -> Self:
        fracture_location, threshold, variables = cls._extract(raw_data)
        return cls(fracture_location, threshold, variables)

    @staticmethod
    def _extract(raw_data):
        fracture_location, threshold = (
            raw_data["strain_values"][k] for k in ("xf", "threshold")
        )
        return fracture_location, threshold, raw_data["strain_diagnostic"]


@attrs.frozen
class SimpleExampleLoader(Loader):
    label: ClassVar[FigureMatcher] = FigureMatcher.SIMPLE_EXAMPLE
    flexural_length: float
    nondim: np.ndarray
    jumps: np.ndarray
    variables: dict

    @classmethod
    def from_raw_data(cls, raw_data) -> Self:
        flexural_length, nondim, jumps, variables = cls._extract(raw_data)
        variables = cls._clean(variables)
        return cls(flexural_length, nondim, jumps, variables)

    @staticmethod
    def _extract(raw_data):
        flexural_length = raw_data["parameters"]["varnish"]["flexural_length"]
        nondim = flexural_length * raw_data["results"]["wavenumbers"]
        variables = {
            k: raw_data["results"][k]
            for k in (
                "relaxation_lengths",
                "amplitude_thresholds",
                "curvature_thresholds",
                "normalised_fractures",
            )
        }
        return flexural_length, nondim, raw_data["jumps"], variables

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
class QuadRegionsLoader(Loader):
    label: ClassVar[FigureMatcher] = FigureMatcher.QUAD_REGIONS
    nondim: np.ndarray
    normalised_fractures: np.ndarray
    variables: list[dict[str, np.ndarray]]

    @classmethod
    def from_raw_data(cls, raw_data) -> Self:
        return cls(*cls._extract(raw_data))

    @staticmethod
    def _extract(
        raw_data,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
        nondim, normalised_fractures = (
            raw_data[k] for k in ("nondim_wavenumbers", "normalised_fractures")
        )
        return nondim, normalised_fractures, [raw_data[f"quad_{i}"] for i in range(4)]


@attrs.frozen
class QuadRegionsDispLoader(Loader):
    label: ClassVar[FigureMatcher] = FigureMatcher.QUAD_REGIONS_DISP
    nondim: np.ndarray
    normalised_fractures: np.ndarray
    relaxation_lengths: np.ndarray
    variables: list[dict[str, np.ndarray]]

    @classmethod
    def from_raw_data(cls, raw_data) -> Self:
        nondim, normalised_fractures, relaxation_lengths, variables = cls._extract(
            raw_data
        )
        normalised_fractures = cls._clean_norm_fractures(normalised_fractures)
        return cls(nondim, normalised_fractures, relaxation_lengths, variables)

    @staticmethod
    def _extract(
        raw_data,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
        short_arrays = (
            raw_data[k]
            for k in (
                "nondim_wavenumbers",
                "normalised_fractures",
                "normalised_relax_lengths",
            )
        )
        return *short_arrays, [raw_data[f"quad_disp_{i}"] for i in range(4)]

    @staticmethod
    def _clean_norm_fractures(normalised_fractures: np.ndarray) -> np.ndarray:
        mask = normalised_fractures > 0.5
        normalised_fractures[mask] = 1 - normalised_fractures[mask]
        return normalised_fractures


@attrs.frozen
class JointDensityLoader(Loader):
    label: ClassVar[FigureMatcher] = FigureMatcher.JOINT_DENSITY
    thicknesses: np.ndarray
    youngs_moduli: np.ndarray

    @classmethod
    def from_raw_data(cls, raw_data) -> Self:
        raw_data = next(iter(raw_data.values()))
        _h, _Y = (raw_data[k] for k in ("thicknesses", "youngs_moduli"))
        return cls(_h, _Y)
