from __future__ import annotations

import abc
import importlib
import pathlib
import typing

import attrs
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from .data import Loader
from .params import GR, WIDTH_TWO_COLUMNS
from .utils import FigureMatcher

# import numpy.typing as npt


def set_style():
    print(pathlib.Path.cwd())
    path_to_style = importlib.resources.files("ff1d_ftsw_pub.lib").joinpath(
        "paper.mplstyle"
    )
    plt.style.use(path_to_style)


@attrs.define
class AbstractPlotter(abc.ABC):
    # label: typing.ClassVar[str]
    figure_number: int
    data: Loader
    size: tuple[float, float]
    figure: Figure = attrs.field(init=False)
    axes: Axes | typing.Sequence[Axes] = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.figure, self.axes = self._init_figure()

    @property
    def filename_handle(self):
        return f"fig{self.figure_number:02d}.pdf"

    @classmethod
    def from_label(cls, label: str, **kwargs) -> AbstractPlotter:
        data_interface = Loader.from_label(label)
        number = getattr(FigureMatcher, label.upper())
        if label == "simple_example":
            size = WIDTH_TWO_COLUMNS, WIDTH_TWO_COLUMNS / GR * 3.481
            return SimpleExamplePlotter(number, data_interface, size, **kwargs)
        else:
            raise NotImplementedError

    def plot(self):
        self._plot()
        self._plot_accessories()
        self._label()
        self._finalise()

    @abc.abstractmethod
    def _init_figure(self): ...

    @abc.abstractmethod
    def _plot(self): ...

    @abc.abstractmethod
    def _plot_accessories(self): ...

    @abc.abstractmethod
    def _label(self): ...

    @abc.abstractmethod
    def _finalise(self): ...

    def _make_filename_handle(self): ...

    def _make_save_path(self): ...

    def _save(self): ...


class SimpleExamplePlotter(AbstractPlotter):
    def _init_figure(self):
        fig, axes = plt.subplots(4, figsize=self.size, sharex=True)
        return fig, axes

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
        lw = plt.rcParams["lines.linewidth"]
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

    def save(self, output_dir):
        fig_name = "fig04.pdf"
        output_dir = self.make_path(output_dir)
        self.figure.savefig(output_dir / fig_name, bbox_inches="tight")
        plt.close()

    def __call__(self):
        return self.plot()


def plot(data_path, plotter): ...
