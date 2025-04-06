import abc
import importlib
import pathlib
import tomllib

import attrs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .params import GR, WIDTH_TWO_COLUMNS


def set_style():
    print(pathlib.Path.cwd())
    path_to_style = importlib.resources.files("ff1d_ftsw_pub.lib").joinpath(
        "paper.mplstyle"
    )
    plt.style.use(path_to_style)


def load_data(subdir: pathlib.Path):
    manifest_name = "manifest.toml"
    with open(subdir.joinpath(manifest_name), "rb") as file:
        manifest = tomllib.load(file)

    return {
        pathlib.Path(handle).stem: getattr(FileFormat, file_format)(
            subdir.joinpath(handle)
        )
        for file_format in manifest
        for handle in manifest[file_format]["files"]
    }


# TODO: actually, maybe want separate classes for reading and plotting? Not sure
@attrs.frozen
class AbstractPlotter(abc.ABC):
    figure_number: int
    # TODO: attrs' factory to build this field from the previous one, with a
    # call to importlib
    data_subdir: pathlib.Path

    @abc.abstractmethod
    # TODO: load_data can probably live here
    def data_loader(self): ...

    @abc.abstractmethod
    def clean_data(self): ...

    @abc.abstractmethod
    def plot(self): ...


# TODO: attrs' attributes like data_root, figname,...
class Fig04(AbstractPlotter):
    def read(
        self,
    ) -> dict[str, npt.ArrayLike | dict[str, npt.ArrayLike]]:
        # TODO: rewrite to get the filenames out
        root = importlib.resources.files("ff1d_ftsw_pub.data").joinpath(
            "fig04"
        )  # pathlib.Path("../data/fig04")
        parameters_file = "reference_experiment_parameters.json"
        with open(root / parameters_file) as f:
            parameters = json.load(f)
        results_file = "stationnary_simple_comparison.npz"
        results = np.load(root / results_file)
        jumps_file = "jumps.csv"
        jumps = np.loadtxt(root / jumps_file, delimiter=",")

        nondim = parameters["varnish"]["flexural_length"] * results["wavenumbers"]
        variables = {"nondim": nondim, "jumps": jumps} | {
            k: results[k]
            for k in (
                "relaxation_lengths",
                "amplitude_thresholds",
                "curvature_thresholds",
                "normalised_fractures",
            )
        }
        return variables

    def clean_fracture_locations(self, variables):
        _y = variables["normalised_fractures"]
        _m = _y > 0.5
        _y[_m] = 1 - _y[_m]
        variables["normalised_fractures"] = _y
        return variables

    def clean_curvature(self, variables):
        variables["curvature_thresholds"] = np.abs(variables["curvature_thresholds"])
        return variables

    # TODO: that does not belong in the class
    def make_path(self, output_dir):
        output_dir = pathlib.Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        return output_dir

    def make_triangle_coordinates(self, nondim, jumps, amplitudes, ratio=3):
        # `ratio` gives how many centred triangles would fit, in logspace,
        # between the two jumps.
        slope = -2
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

    def add_triangle(self, ax, variables):
        lw = 1
        fontsize = "xx-small"
        triangle_coordinates = self.make_triangle_coordinates(
            variables["nondim"], variables["jumps"], variables["amplitude_thresholds"]
        )
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

    def plot_fracture_loc(self, ax, jumps, nondim, fracture_location, lw):
        bounds = 0, *jumps, np.inf
        for lb, ub in zip(bounds[:-1], bounds[1:]):
            mask = (nondim >= lb) & (nondim < ub)
            ax.semilogx(nondim[mask], fracture_location[mask], "C3", lw=lw)
        ax.set_ylabel("Fracture location")

        horizontal_asymptotes = 1 / 6, 1 / 3, 1 / 2
        for val in horizontal_asymptotes:
            ax.axhline(val, c="k", lw=lw / 3, ls="--")

    def plot_others(self, ax, jumps, nondim, variable, ylabel, lw):
        ax.semilogx(nondim, variable, "C3", lw=lw)
        ax.set_ylabel(ylabel)
        for jump in jumps:
            ax.axvline(jump, c="k", lw=lw / 3)

    def plot(self):
        lw = 1
        width = WIDTH_TWO_COLUMNS
        height = width / GR * 3.481
        variables = self.clean_fracture_locations(self.read())
        variables = self.clean_curvature(variables)

        fig, axes = plt.subplots(4, dpi=300, figsize=(width, height), sharex=True)
        self.plot_fracture_loc(
            axes[0],
            variables["jumps"],
            variables["nondim"],
            variables["normalised_fractures"],
            lw,
        )
        ylabels = (
            "Crit. amplitude (m)",
            "Crit. curvature (m$^{-1}$)",
            "Relaxation length (m)",
        )
        for ax, variable, ylabel in zip(
            axes[1:],
            (
                variables["amplitude_thresholds"],
                variables["curvature_thresholds"],
                variables["relaxation_lengths"],
            ),
            ylabels,
        ):
            self.plot_others(
                ax, variables["jumps"], variables["nondim"], variable, ylabel, lw
            )
        self.add_triangle(axes[1], variables)

        axes[1].set_yscale("log")
        axes[-1].set_xlabel("$k L_D$")
        axes[-1].set_xlim(0.1, 2.5)

        fig.tight_layout()
        # print([ax.bbox.width / ax.bbox.height for ax in axes])
        return fig

    def save(self, output_dir):
        fig_name = "fig04.pdf"
        output_dir = self.make_path(output_dir)
        fig = self.plot()
        fig.savefig(output_dir / fig_name, bbox_inches="tight")


def plot(data_path, plotter): ...
