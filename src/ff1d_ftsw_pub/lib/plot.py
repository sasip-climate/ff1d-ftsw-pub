from __future__ import annotations

import abc
import importlib
import pathlib
import typing

import attrs
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data import Loader
from .params import GR, WIDTH_SINGLE_COLUMN, WIDTH_TWO_COLUMNS
from .utils import FigureMatcher

# import numpy.typing as npt

plotters_registry = dict()
figure_sizes = {
    FigureMatcher.SIMPLE_EXAMPLE: (WIDTH_TWO_COLUMNS, WIDTH_TWO_COLUMNS / GR * 3.481),
    FigureMatcher.JOINT_DENSITY: (None, WIDTH_TWO_COLUMNS),
}


def set_style():
    path_to_style = importlib.resources.files("ff1d_ftsw_pub.lib").joinpath(
        "paper.mplstyle"
    )
    plt.style.use(path_to_style)


@attrs.define
class AbstractPlotter(abc.ABC):
    label: typing.ClassVar[FigureMatcher]
    figure_number: int
    data: Loader
    size: tuple[float, float | None]
    figure: Figure = attrs.field(init=False)
    axes: Axes | typing.Sequence[Axes] = attrs.field(init=False)

    def __attrs_post_init__(self):
        self._init_figure_axes()

    @classmethod
    def __attrs_init_subclass__(cls):
        plotters_registry[cls.label] = cls

    @property
    def filename_handle(self):
        return f"fig{self.figure_number:02d}.pdf"

    @classmethod
    def from_label(cls, label: FigureMatcher, **kwargs) -> AbstractPlotter:
        data_interface = Loader.from_label(label)
        number = label.value
        figsize = figure_sizes[label]
        return plotters_registry[label](number, data_interface, figsize, **kwargs)

    def _plot_wrapper(self):
        self._plot()
        self._plot_accessories()
        self._label()
        self._finalise()

    def _init_figure(self):
        self.figure = Figure(figsize=self.size)

    @abc.abstractmethod
    def _init_axes(self): ...

    def _init_figure_axes(self):
        self._init_figure()
        self._init_axes()

    @abc.abstractmethod
    def _plot(self): ...

    @abc.abstractmethod
    def _plot_accessories(self): ...

    @abc.abstractmethod
    def _label(self): ...

    @abc.abstractmethod
    def _finalise(self): ...

    def _save(self, output_dir: pathlib.Path):
        self.figure.savefig(
            output_dir.joinpath(self.filename_handle),
            dpi=self.figure.dpi,
            bbox_inches="tight",
        )

    def make_and_write(self, output_dir):
        self._plot_wrapper()
        self._save(output_dir)
        plt.close(self.figure)


@attrs.define
class SimpleExamplePlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.SIMPLE_EXAMPLE

    def _init_axes(self):
        self.axes = self.figure.subplots(4, sharex=True)

    def _label(self):
        ylabels = (
            "Fracture location",
            "Crit. amplitude (m)",
            "Crit. curvature (m$^{-1}$)",
            "Relaxation length (m)",
        )
        for ax, label in zip(self.axes, ylabels):
            ax.set_ylabel(label)
        self.axes[-1].set_xlabel("$k L_D$")

    def _plot(self):
        self._plot_fracture_loc()
        self._plot_others()

    def _plot_fracture_loc(self):
        ax = self.axes[0]
        nondim = self.data.nondim
        bounds = 0, *self.data.jumps, np.inf
        for lb, ub in zip(bounds[:-1], bounds[1:]):
            mask = (nondim >= lb) & (nondim < ub)
            ax.semilogx(
                nondim[mask],
                self.data.variables["normalised_fractures"][mask],
                "C3",
            )
        ax.set_ylabel("Fracture location")

    def _plot_others(self):
        for ax, variable_key in zip(
            self.axes[1:],
            (
                "amplitude_thresholds",
                "curvature_thresholds",
                "relaxation_lengths",
            ),
        ):
            ax.semilogx(self.data.nondim, self.data.variables[variable_key], "C3")

    def _plot_accessories(self):
        self._plot_fracture_loc_asymptotes()
        self._add_triangle(ratio=3)
        self._add_jumps()

    def _plot_fracture_loc_asymptotes(self):
        lw = plt.rcParams["lines.linewidth"] / 3
        horizontal_asymptotes = 1 / 6, 1 / 3, 1 / 2
        ax = self.axes[0]
        for val in horizontal_asymptotes:
            ax.axhline(val, c="k", lw=lw, ls="--")

    def _make_triangle_coordinates(self, ratio):
        # `ratio` gives how many centred triangles would fit, in logspace,
        # between the two jumps.
        slope = -2
        nondim, jumps, amplitudes = (
            self.data.nondim,
            self.data.jumps,
            self.data.variables["amplitude_thresholds"],
        )
        jump_idx = np.argmin(np.abs(nondim - jumps[1]))
        # Triangle in region 2, between jumps[0] and jumps[1]. We align the
        # base with the intersection of jumps[1] and the critical amplitude
        # curve.
        base_triangle_y = amplitudes[jump_idx]
        base_triangle_x0, base_triangle_x1 = np.exp(
            np.linalg.solve(
                [[-1, 1], [1, 1]],
                [
                    np.log(jumps[1] / jumps[0]) / ratio,
                    np.log(jumps[1] * jumps[0]),
                ],
            )
        )
        top_triangle_y = (
            base_triangle_y * (base_triangle_x0 / base_triangle_x1) ** slope
        )
        return np.array(
            (
                (base_triangle_x0, base_triangle_y),
                (base_triangle_x1, base_triangle_y),
                (base_triangle_x0, top_triangle_y),
            )
        )

    def _add_triangle(self, ratio):
        lw = plt.rcParams["lines.linewidth"] * 2 / 3
        ax = self.axes[1]
        fontsize = "xx-small"
        triangle_coordinates = self._make_triangle_coordinates(ratio)
        ax.add_patch(
            plt.Polygon(
                triangle_coordinates,
                facecolor="#0000",
                edgecolor="C0",
                lw=lw,
            )
        )
        ax.text(
            np.sqrt(triangle_coordinates[0, 0] * triangle_coordinates[1, 0]),
            triangle_coordinates[0, 1] - 2e-4,
            "1",
            ha="center",
            va="top",
            fontsize=fontsize,
        )
        ax.text(
            triangle_coordinates[0, 0] - 4e-3,
            np.sqrt(triangle_coordinates[0, 1] * triangle_coordinates[2, 1]),
            "2",
            ha="right",
            va="center",
            fontsize=fontsize,
        )

    def _add_jumps(self):
        lw = plt.rcParams["lines.linewidth"] / 3
        for ax in self.axes[1:]:
            for jump in self.data.jumps:
                ax.axvline(jump, c="k", lw=lw)

    def _finalise(self):
        self.axes[1].set_yscale("log")
        self.axes[-1].set_xlim(0.1, 2.5)
        self.figure.tight_layout()

    def __call__(self):
        return self._plot_wrapper()


@attrs.define
class JointDensityPlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.JOINT_DENSITY
    joint_grid: sns.JointGrid = attrs.field(init=False)

    def _init_figure(self):
        self.joint_grid = sns.JointGrid(
            x=self._metre_to_micrometre(self.data.thicknesses),
            y=self._pascal_to_megapascal(self.data.youngs_moduli),
            height=self.size[1],
            ratio=4,
        )
        self.figure = self.joint_grid.figure

    def _init_axes(self):
        self.axes = self.figure.axes

    def _plot(self):
        self.joint_grid.plot_joint(sns.scatterplot, c="C3", alpha=0.75)
        hist_facecolor = mpl.colors.to_rgba("C3", alpha=0.25)
        self.joint_grid.plot_marginals(
            sns.histplot, edgecolor=None, facecolor=hist_facecolor
        )

    @staticmethod
    def _metre_to_micrometre(var: np.ndarray) -> np.ndarray:
        return var * 1e6

    @staticmethod
    def _pascal_to_megapascal(var: np.ndarray) -> np.ndarray:
        return var / 1e6

    def _plot_accessories(self):
        pass

    def _label(self):
        self.joint_grid.set_axis_labels("Thickness (µm)", "Young's modulus (MPa)")

    def _finalise(self):
        self.figure.tight_layout()


def plot(data_path, plotter): ...
