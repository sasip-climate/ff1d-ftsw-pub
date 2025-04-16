from __future__ import annotations

import abc
import importlib
import itertools
import pathlib
import typing

import attrs
import cmocean.cm as cmo
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from .data import Loader
from .params import GR, WIDTH_SINGLE_COLUMN, WIDTH_TWO_COLUMNS
from .utils import FigureMatcher

# import numpy.typing as npt
thin_line = dict(lw=plt.rcParams["lines.linewidth"] / 3, c="k")
thin_dashed_line = thin_line | dict(ls="--")

plotters_registry = dict()
figure_sizes = {
    FigureMatcher.SCHEMATICS: (5.4, 2.6),
    FigureMatcher.FRACTURE_SEARCH: (
        0.8 * WIDTH_SINGLE_COLUMN,
        WIDTH_SINGLE_COLUMN / 1.224,
    ),
    FigureMatcher.STRAIN_FRACTURE: (WIDTH_TWO_COLUMNS, WIDTH_TWO_COLUMNS / GR / 0.86),
    FigureMatcher.SIMPLE_EXAMPLE: (WIDTH_TWO_COLUMNS, WIDTH_TWO_COLUMNS / GR * 3.481),
    FigureMatcher.QUAD_REGIONS: (WIDTH_SINGLE_COLUMN, WIDTH_SINGLE_COLUMN / GR / 0.940),
    FigureMatcher.QUAD_REGIONS_DISP: (
        WIDTH_SINGLE_COLUMN,
        WIDTH_SINGLE_COLUMN / GR / 0.860,
    ),
    FigureMatcher.JOINT_DENSITY: (None, WIDTH_TWO_COLUMNS),
    FigureMatcher.ENS_AMPLITUDE: (WIDTH_SINGLE_COLUMN, WIDTH_SINGLE_COLUMN / GR / 1.01),
    FigureMatcher.ENS_DIMENSIONLESS: (
        WIDTH_TWO_COLUMNS,
        WIDTH_TWO_COLUMNS / GR * 2 / 1.019,
    ),
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
    data: type[Loader]
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
class SchemaPlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.SCHEMATICS

    def _init_axes(self):
        self.axes = self.figure.subplots()

    def _make_polygons(
        self, rest_lw: float, defl_lw: float, colour: str, floe_fact: float
    ):
        floe_fact = 1.25

        deflected_floe_top = self.data.variables["displacement"] + self.data.freeboard
        deflected_floe_bottom = self.data.variables["displacement"] - self.data.draught

        deflected_floe = Polygon(
            np.vstack(
                (
                    np.hstack(
                        (
                            self.data.variables["along_floe_axis"],
                            self.data.variables["along_floe_axis"][::-1],
                        )
                    ),
                    np.hstack((deflected_floe_top, deflected_floe_bottom[::-1])),
                )
            ).T,
            closed=True,
            facecolor="#0000",
            edgecolor=colour,
            lw=defl_lw * floe_fact,
            label="Deflected floe",
        )
        at_rest_floe = Polygon(
            [
                [self.data.variables["along_floe_axis"][0], self.data.freeboard],
                [self.data.variables["along_floe_axis"][-1], self.data.freeboard],
                [self.data.variables["along_floe_axis"][-1], -self.data.draught],
                [self.data.variables["along_floe_axis"][0], -self.data.draught],
            ],
            closed=True,
            facecolor="#0000",
            edgecolor=colour,
            ls="--",
            lw=rest_lw * floe_fact,
            label="Floe at rest",
        )

        return at_rest_floe, deflected_floe

    def _plot(self):
        rest_lw = 0.65
        defl_lw = 0.85
        floe_fact = 1.25
        fluid_colour = cmo.ice(0.05)
        floe_colour = cmo.ice(0.618)
        self._plot_fluid(rest_lw, defl_lw, fluid_colour)
        self._plot_floe(rest_lw, defl_lw, floe_colour, floe_fact)

    def _plot_fluid(
        self, rest_lw: float, defl_lw: float, fluid_colour: tuple[float, ...]
    ):
        ax = self.axes
        ax.plot(
            (
                self.data.variables["pre_floe_axis"][0],
                self.data.variables["post_floe_axis"][-1],
            ),
            (0, 0),
            c=fluid_colour,
            ls="--",
            lw=rest_lw,
            label="Fluid at rest",
        )
        ax.plot(
            self.data.variables["along_floe_axis"],
            self.data.variables["along_floe_surface"],
            lw=defl_lw,
            c=fluid_colour,
        )
        ax.plot(
            self.data.variables["pre_floe_axis"],
            self.data.variables["pre_floe_surface"],
            lw=defl_lw,
            c=fluid_colour,
        )
        ax.plot(
            self.data.variables["post_floe_axis"],
            self.data.variables["post_floe_surface"],
            lw=defl_lw,
            c=fluid_colour,
            label=r"$\eta(x)$",
        )

    def _plot_floe(self, rest_lw, defl_lw, floe_colour, floe_fact):
        ax = self.axes
        at_rest_floe, deflected_floe = self._make_polygons(
            rest_lw, defl_lw, floe_colour, floe_fact
        )
        ax.add_patch(at_rest_floe)
        ax.add_patch(deflected_floe)

    def _plot_accessories(self):
        text_dict = dict(
            fontsize="small",
        )
        self._add_arrows(text_dict)
        self._annotate(text_dict)

    def _make_arrows(self):
        arrow_dict = dict(
            arrowstyle="<->",
            lw=0.35,
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
        )

        idx = 74
        thickness_arrow = FancyArrowPatch(
            (self.data.variables["along_floe_axis"][idx], self.data.freeboard),
            (self.data.variables["along_floe_axis"][idx], -self.data.draught),
            **arrow_dict,
        )
        idx = 192
        perturbation_arrow = FancyArrowPatch(
            (
                self.data.variables["along_floe_axis"][idx],
                self.data.variables["along_floe_surface"][idx],
            ),
            (
                self.data.variables["along_floe_axis"][idx],
                self.data.variables["displacement"][idx] - self.data.draught,
            ),
            **arrow_dict,
        )
        return thickness_arrow, perturbation_arrow

    def _add_arrows(self, text_dict):
        arrows = self._make_arrows()
        for arrow in arrows:
            self.axes.add_patch(arrow)
        self._annotate_arrows(arrows, text_dict)

    def _annotate_arrows(self, arrows: tuple[FancyArrowPatch, ...], text_dict):
        thickness_arrow, perturbation_arrow = arrows
        ax: Axes = self.axes
        ax.text(
            thickness_arrow._posA_posB[0][0] + 1,
            (self.data.freeboard - self.data.draught) / 2,
            "$h$",
            horizontalalignment="left",
            verticalalignment="center",
            **text_dict,
        )
        ax.text(
            perturbation_arrow._posA_posB[0][0] + 1,
            (
                2 * perturbation_arrow._posA_posB[0][1]
                + perturbation_arrow._posA_posB[1][1]
            )
            / 3,
            r"$\eta(x) - [w(x) - d]$",
            horizontalalignment="left",
            verticalalignment="center",
            **text_dict,
        )

    def _annotate(self, text_dict):
        arrowprops = dict(lw=0.5, ls=(0, (1, 5)), arrowstyle="-")
        deflected_floe_bottom = self.data.variables["displacement"] - self.data.draught

        ax = self.axes
        ax.annotate(
            "$z=0$",
            (
                self.data.variables["pre_floe_axis"][0],
                0,
            ),
            (
                self.data.variables["pre_floe_axis"][0] - 5,
                0,
            ),
            horizontalalignment="right",
            verticalalignment="center",
            arrowprops=arrowprops,
            **text_dict,
        )
        ax.annotate(
            "$z=-d$",
            (self.data.variables["along_floe_axis"][0], -self.data.draught),
            (self.data.variables["pre_floe_axis"][0] - 5, -self.data.draught),
            horizontalalignment="right",
            verticalalignment="center",
            arrowprops=arrowprops,
            **text_dict,
        )
        ax.annotate(
            "x=0",
            (
                self.data.variables["along_floe_axis"][0],
                deflected_floe_bottom[0],
            ),
            (
                self.data.variables["along_floe_axis"][0],
                deflected_floe_bottom.min() - 0.1,
            ),
            horizontalalignment="center",
            verticalalignment="bottom",
            arrowprops=arrowprops,
            **text_dict,
        )
        ax.annotate(
            "x=L",
            (self.data.variables["along_floe_axis"][-1], -self.data.draught),
            (
                self.data.variables["along_floe_axis"][-1],
                deflected_floe_bottom.min() - 0.1,
            ),
            horizontalalignment="center",
            verticalalignment="bottom",
            arrowprops=arrowprops,
            **text_dict,
        )

    def _label(self): ...

    def _finalise(self):
        ax = self.axes
        ax.set_ylim(-0.7, 0.3)
        ax.set_aspect(5e1)
        ax.set_axis_off()

        ax.legend(
            ncols=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            # fontsize="small",
            handlelength=1.25,
            handletextpad=0.5,
        )
        self.figure.tight_layout()


@attrs.define
class FractureSearchPlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.FRACTURE_SEARCH

    def _init_axes(self):
        self.axes = self.figure.subplots(2, sharex=True, height_ratios=(1, 1 / GR))

    def _make_polygons(self, lw):
        along_floe_axis = self.data.deflection_variables["pref_x"]
        deflected_floe_top = (
            self.data.deflection_variables["pref_displacement"] + self.data.freeboard
        )
        deflected_floe_bottom = (
            self.data.deflection_variables["pref_displacement"] - self.data.draught
        )
        deflected_floe = Polygon(
            np.vstack(
                (
                    np.hstack((along_floe_axis, along_floe_axis[::-1])),
                    np.hstack((deflected_floe_top, deflected_floe_bottom[::-1])),
                )
            ).T,
            closed=True,
            facecolor="#0000",
            edgecolor="C3",
            lw=lw,
        )

        postf_along_floe_axes = [
            self.data.deflection_variables[k] for k in ("postf_x_left", "postf_x_right")
        ]
        postf_deflected_floe_tops = [
            self.data.deflection_variables[k] + self.data.freeboard
            for k in ("postf_displacement_left", "postf_displacement_right")
        ]
        postf_deflected_floe_bottoms = [
            self.data.deflection_variables[k] - self.data.draught
            for k in ("postf_displacement_left", "postf_displacement_right")
        ]
        postf_deflected_floes = [
            Polygon(
                np.vstack(
                    (np.hstack((_x, _x[::-1])), np.hstack((_df_top, _df_bottom[::-1])))
                ).T,
                closed=True,
                facecolor="#0000",
                edgecolor=f"C{i}",
                lw=lw,
            )
            for i, (_x, _df_top, _df_bottom) in enumerate(
                zip(
                    postf_along_floe_axes,
                    postf_deflected_floe_tops,
                    postf_deflected_floe_bottoms,
                )
            )
        ]
        return deflected_floe, *postf_deflected_floes

    def _plot(self):
        lw = 1.5
        self._plot_energy(lw)
        self._plot_floes(lw)

    def _plot_energy(self, lw):
        ax = self.axes[0]
        x = self.data.energy_variables["x"]
        energy = self.data.energy_variables["energy"]
        ax.plot(x, energy[:, 0], lw=lw, label="Left fragment")
        ax.plot(x, energy[:, 1], lw=lw, label="Right fragment")
        ax.plot(
            x,
            energy.sum(axis=1) + self.data.energy_scalars["energy_release_rate"],
            label="Post-fracture total",
            ls="--",
            lw=lw,
        )
        ax.axhline(
            self.data.energy_scalars["initial_energy"],
            label="Initial floe",
            c="C3",
            lw=lw,
        )

    def _plot_floes(self, lw):
        floes = self._make_polygons(lw)
        ax = self.axes[1]
        for patch in floes:
            ax.add_patch(patch)

    def _plot_accessories(self):
        self._plot_fracture_loc()
        self._plot_relax_length()

    def _plot_fracture_loc(self):
        self.axes[0].axvline(self.data.energy_scalars["xf"], lw=0.5, zorder=-10, c="k")

    def _plot_relax_length(self):
        xf = self.data.energy_scalars["xf"]
        relaxation_length = self.data.energy_scalars["relaxation_length"]
        lb, ub = xf + np.array((-relaxation_length, relaxation_length)) / 2
        self.axes[1].axvspan(lb, ub, alpha=0.1, facecolor="C3", zorder=-10)

    def _label(self):
        ax = self.axes[0]
        ax.set_xlabel("Along-floe coordinate of a hypothetical fracture (m)")
        ax.set_ylabel("Energy (J m$^{-2}$)")

        ax = self.axes[1]
        ax.set_xlabel("Horizontal coordinate (m)")
        ax.set_ylabel("Vertical coordinate (m)")

        self.figure.legend(
            ncols=4,
            handlelength=1.85,
            loc="upper center",
            bbox_to_anchor=(
                (self.axes[0].bbox.x0 + self.axes[0].bbox.x1)
                / 2
                / self.figure.bbox.width,
                1.05,
            ),
            handletextpad=0.33,
            columnspacing=1,
        )

    def _finalise(self):
        bounds = self.data.energy_variables["x"][[0, -1]]
        self.axes[0].set_xlim(*bounds)
        self.axes[1].set_ylim(-0.7, 0.3)
        self.figure.tight_layout()


@attrs.define
class StrainFracturePlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.STRAIN_FRACTURE

    def _init_axes(self):
        self.axes = self.figure.subplots()

    def _plot(self):
        self.axes.plot(
            self.data.variables["x"],
            self.data.variables["strain"],
            "C3",
        )

    def _highlight_peaks(self):
        self.axes.scatter(
            self.data.variables["peaks"],
            self.data.variables["extrema"],
            c="C1",
            marker="+",
            zorder=10,
        )

    def _plot_fracture_loc(self):
        self.axes.axvline(self.data.fracture_location, zorder=-10, c="k", lw=0.5)

    def _make_exceeding_boundaries(self):
        strain_exceeds_threshold = np.nonzero(
            np.abs(self.data.variables["strain"]) > self.data.threshold
        )[0]
        return np.vstack(
            (
                np.hstack(
                    (
                        strain_exceeds_threshold[0],
                        strain_exceeds_threshold[
                            np.nonzero(np.ediff1d(strain_exceeds_threshold) > 1)[0] + 1
                        ],
                    )
                ),
                np.hstack(
                    (
                        strain_exceeds_threshold[
                            np.nonzero(np.ediff1d(strain_exceeds_threshold) > 1)[0]
                        ],
                        strain_exceeds_threshold[-1],
                    )
                ),
            )
        ).T

    def _add_exceending_strain_bounds(self):
        exceeding_strain_boundaries = self._make_exceeding_boundaries()
        for _p in exceeding_strain_boundaries:
            self.axes.axvspan(*self.data.variables["x"][_p], facecolor="C3", alpha=0.1)

    def _plot_accessories(self):
        self._highlight_peaks()
        self._plot_fracture_loc()
        self._add_exceending_strain_bounds()

    def _label(self):
        self.axes.set_xlabel("Horizontal coordinate (m)")
        self.axes.set_ylabel("Strain")

    def _finalise(self):
        ax = self.axes
        ax.set_xlim(self.data.variables["x"][[0, -1]])
        ax.set_ylim(-2.5e-4, 2.5e-4)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        self.figure.tight_layout()


@attrs.define
class SimpleExamplePlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.SIMPLE_EXAMPLE
    xlim: tuple[float, float] = 0.1, 2.5

    def _init_axes(self):
        self.axes = self.figure.subplots(4, sharex=True)

    def _label(self):
        ylabels = (
            "Normalised\nfracture location",
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
        self._plot_fracture_loc_asymptotes(thin_dashed_line)
        self._add_triangle(ratio=3)
        self._add_jumps(thin_line)
        self._add_region_markers()
        self._add_relaxation_asymptote(thin_dashed_line)

    def _plot_fracture_loc_asymptotes(self, params):
        horizontal_asymptotes = 1 / 6, 1 / 3, 1 / 2
        ax = self.axes[0]
        for val in horizontal_asymptotes:
            ax.axhline(val, **params)

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

    def _add_jumps(self, params):
        for ax in self.axes[1:]:
            for jump in self.data.jumps:
                ax.axvline(jump, **params)

    def _add_region_markers(self):
        boundaries = np.array((self.xlim[0], *self.data.jumps, self.xlim[1]))
        middles = np.sqrt(boundaries[:-1] * boundaries[1:])
        # Assuming curvature plot
        _y_coord = 6
        _y_coord_alt = 8  # to avoid the curve in region 4
        for region_num, x_coord in zip(range(1, 5), middles):
            y_coord = _y_coord if region_num <= 3 else _y_coord_alt
            self.axes[2].annotate(
                f"{region_num}",
                (x_coord, y_coord),
                bbox={"boxstyle": "circle", "facecolor": "#0000", "pad": 0.1},
                ha="center",
                va="center",
                fontsize="x-small",
            )

    def _add_relaxation_asymptote(self, params):
        self.axes[3].axhline(self.data.flexural_length * 2**0.5, **params)

    def _finalise(self):
        self.axes[1].set_yscale("log")
        self.axes[-1].set_xlim(self.xlim)
        self.figure.tight_layout()

    def __call__(self):
        return self._plot_wrapper()


@attrs.define
class QuadRegionsPlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.QUAD_REGIONS
    twin_axes: typing.Sequence[Axes] = attrs.field(init=False)

    def _init_axes(self):
        self.axes = self.figure.subplots(2, 2, sharex=True, sharey=True)
        self.twin_axes = [ax.twinx() for ax in np.ravel(self.axes)]
        for tax0, tax1 in zip(self.twin_axes[:-1], self.twin_axes[1:]):
            tax0.sharey(tax1)

    def _plot(self):
        lw = 1
        self._plot_free_energy()
        self._plot_curvatures()

    def _plot_free_energy(self):
        for i, ax in enumerate(np.ravel(self.axes)):
            ax.plot(
                self.data.variables[i]["x"],
                self.data.variables[i]["free_energy"],
                "C2",
                label=r"$F(x_1)$",
            )

    def _plot_curvatures(self):
        for i, tax in enumerate(np.ravel(self.twin_axes)):
            tax.plot(
                self.data.variables[i]["x"],
                self.data.variables[i]["curvature"],
                "C3",
                label=r"$\kappa(x)$",
            )
            tax.plot(
                self.data.variables[i]["x"],
                self.data.variables[i]["conforming_curvature"],
                "C4--",
                label=r"$\kappa_{\mathrm{conf}}(x)$",
            )

    def _plot_accessories(self):
        self._add_fracture_locations()
        self._add_zero_curvature()

    def _add_fracture_locations(self):
        for i, ax in enumerate(np.ravel(self.axes)):
            xf = self.data.normalised_fractures[i]
            ax.axvline(xf, **thin_line)
            if not np.isclose(xf - 0.5, 0):
                ax.axvline(1 - xf, **thin_line)

    def _add_zero_curvature(self):
        for tax in np.ravel(self.twin_axes):
            tax.axhline(0, **thin_dashed_line)

    def _label(self):
        axes, twin_axes = self.axes, self.twin_axes
        for i, ax in enumerate(np.ravel(axes)):
            ax.set_title(f"Region {i + 1}, $k L_D = {{{self.data.nondim[i]:1.4f}}}$")
        for ax in axes[1, :]:
            ax.set_xlabel("Normalised coordinate")
        for ax in axes[:, 0]:
            ax.set_ylabel("Free energy (J m$^{-2}$)")
        for tax in twin_axes[1::2]:
            tax.set_ylabel("Curvature (m$^{-1}$)")

        handles, labels = zip(
            ax.get_legend_handles_labels(), tax.get_legend_handles_labels()
        )
        handles, labels = (itertools.chain(*_e) for _e in (handles, labels))
        self.figure.legend(
            handles=handles,
            labels=labels,
            ncols=3,
            handlelength=1.8,
            loc="upper center",
            fontsize="small",
            bbox_to_anchor=(0.525, 1.06),
            handletextpad=0.33,
            columnspacing=1.5,
        )

    def _finalise(self):
        for tax in self.twin_axes[::2]:
            plt.setp(tax.get_yticklabels(), visible=False)
        self.figure.tight_layout()


@attrs.define
class QuadRegionsDispPlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.QUAD_REGIONS_DISP

    def _init_axes(self):
        self.axes = self.figure.subplots(2, 2, sharex=True, sharey=True)

    def _plot(self):
        self._plot_deflection()

    def _plot_deflection(self):
        for i, ax in enumerate(np.ravel(self.axes)):
            ax.plot(
                self.data.variables[i]["pref_x"],
                self.data.variables[i]["pref_w"],
                color="C3",
                label="Initial floe",
            )
            ax.plot(
                self.data.variables[i]["postf_x1"],
                self.data.variables[i]["postf_w1"],
                "--",
                label="Left fragment",
            )
            ax.plot(
                self.data.variables[i]["postf_x2"],
                self.data.variables[i]["postf_w2"],
                "--",
                label="Right fragment",
            )

    def _plot_accessories(self):
        self._add_zero_deflection()
        self._add_fracture_locations()
        self._add_relaxation_lengths()

    def _add_zero_deflection(self):
        for ax in np.ravel(self.axes):
            ax.axhline(0, **thin_dashed_line)

    def _add_fracture_locations(self):
        for i, ax in enumerate(np.ravel(self.axes)):
            ax.axvline(self.data.normalised_fractures[i], **thin_line)

    def _add_relaxation_lengths(self):
        for i, ax in enumerate(np.ravel(self.axes)):
            lb, rb = (
                np.array((-1, 1)) * self.data.relaxation_lengths[i] / 2
            ) + self.data.normalised_fractures[i]
            ax.axvspan(lb, rb, alpha=0.2, facecolor="C3")

    def _label(self):
        axes = self.axes
        for i, ax in enumerate(np.ravel(axes)):
            ax.set_title(f"Region {i + 1}, $k L_D = {{{self.data.nondim[i]:1.4f}}}$")
        for ax in axes[1, :]:
            ax.set_xlabel("Normalised coordinate")
        for ax in axes[:, 0]:
            ax.set_ylabel("Normalised deflection")

        handles, labels = ax.get_legend_handles_labels()

        self.figure.legend(
            handles=handles,
            labels=labels,
            ncols=3,
            handlelength=2.4,
            loc="upper center",
            # fontsize="small",
            bbox_to_anchor=(0.525, 1.05),
            handletextpad=0.33,
            columnspacing=1.5,
        )

    def _finalise(self):
        self.figure.tight_layout()


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


@attrs.define
class EnsembleAmplitudePlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.ENS_AMPLITUDE
    twin_ax: Axes = attrs.field(init=False)

    def _init_axes(self):
        self.axes = self.figure.subplots()
        self.twin_ax = self.axes.twiny()
        self.twin_ax.sharex(self.axes)

    def _plot(self):
        self._plot_ensemble()
        self._plot_data()

    def _plot_ensemble(self):
        hue = "flexural_rigidity"
        sns.lineplot(
            self.data.ensemble,
            x="nondim",
            y="amplitude_threshold",
            ax=self.axes,
            hue=hue,
            palette="crest",
            legend="full",
        )
        self._add_colorbar(hue)

    def _plot_data(self):
        sns.scatterplot(
            x=(
                self.data.experimental["wavenumbers"]
                * self.data.experimental["flex_lengths"]
            ),
            y=self.data.experimental["critical_amplitudes"],
            ax=self.axes,
            color="C3",
            zorder=10,
            alpha=0.75,
            label=r"$a_\mathrm{cr}(kL_D)$",
        )
        sns.scatterplot(
            x=self.data.experimental["flex_lengths"] / self.data.experimental["fwhm"],
            y=self.data.experimental["critical_amplitudes"],
            ax=self.twin_ax,
            color="C4",
            marker="s",
            alpha=0.75,
            zorder=8,
            label=r"$a_\mathrm{cr}(L_D / \mathrm{FWHM})$",
        )

    def _plot_accessories(self): ...

    def _add_colorbar(self, hue):
        if hue == "flexural_rigidity":
            label = "Flexural rigidity (Pa m$^{3}$)"
        else:
            raise NotImplementedError
        vmin, vmax = np.sort(
            (self.data.ensemble.select(hue).unique().to_numpy()[:, 0])
        )[[0, -1]]
        self.figure.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False),
                cmap="crest",
            ),
            ax=self.twin_ax,
            label=label,
        )

    def _label(self):
        ax, axt = self.axes, self.twin_ax
        axt.set_xlabel(r"$\frac{L_D}{\mathrm{FWHM}}$")
        ax.set_xlabel("$k L_D$")
        ax.set_ylabel("Critical amplitude (m)")

        ax.get_legend().set_visible(False)
        axt.get_legend().set_visible(False)

        handles, labels = zip(
            list(zip(*ax.get_legend_handles_labels()))[-1],
            next(zip(*axt.get_legend_handles_labels())),
        )
        ax.legend(handles, labels)

    def _finalise(self):
        self.axes.loglog()
        self.figure.tight_layout()


@attrs.define
class EnsembleDimensionlessPlotter(AbstractPlotter):
    label: typing.ClassVar[FigureMatcher] = FigureMatcher.ENS_DIMENSIONLESS

    def _init_axes(self):
        self.axes = self.figure.subplots(2, sharex=True)

    def _plot(self):
        self._plot_ensembles(self._compute_mask())

    def _compute_mask(self):
        return (pl.col("harmonic_number") % 2 == 0) & (pl.col("nondim2") < 1e-16)

    def _plot_ensembles(self, rough_mask):
        palette = sns.color_palette("mako_r", len(self.data.harmonics))
        lineplot_kwds = dict(
            alpha=0.85,
            hue="harmonic_number",
            hue_order=self.data.harmonics,
            palette=palette,
        )

        for results in (
            self.data.ensemble.filter(~rough_mask),
            self.data.ensemble.filter(rough_mask),
        ):
            ax = self.axes[0]
            sns.lineplot(
                results,
                x="nondim",
                y="normalised_dis_length",
                ax=ax,
                legend=False,
                **lineplot_kwds,
            )

            ax = self.axes[1]
            sns.lineplot(
                results,
                x="nondim",
                y="nondim2",
                ax=ax,
                **lineplot_kwds,
            )

    def _plot_accessories(self):
        pass

    def _label(self):
        ax = self.axes[0]
        ax.set_ylabel(r"$\frac{L_{\kappa}}{\sqrt{2} L_D}$")
        ax = self.axes[1]
        ax.set_xlabel("$k L_D$")
        ax.set_ylabel(r"${\kappa_{\text{cr}}}^2 h L_{\kappa}$")

    def _finalise(self):
        number_of_harmonics = len(self.data.harmonics)
        ax = self.axes[-1]
        ax.set_xscale("log")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.get_legend().set_visible(False)
        ax.set_xlim(1e-1, 2.5)

        self.figure.tight_layout()
        _bbox = (
            self.axes[0]
            .get_window_extent()
            .transformed(self.figure.transFigure.inverted())
        )
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = handles[:number_of_harmonics], labels[:number_of_harmonics]
        labels = [f"$n={{{_l}}}$" for _l in labels]
        self.figure.legend(
            handles=handles,
            labels=labels,
            ncols=4,
            handlelength=0.75,
            loc="upper center",
            fontsize="small",
            bbox_to_anchor=((_bbox.x0 + _bbox.x1) / 2, 1.05),
            handletextpad=0.33,
            columnspacing=1.0,
        )
