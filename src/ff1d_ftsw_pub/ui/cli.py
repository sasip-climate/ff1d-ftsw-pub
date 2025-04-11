import argparse
import pathlib

from ..lib import plot
from ..lib.utils import FigureMatcher


def ensure_output_path_exists(output_dir: pathlib.Path):
    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for the publication."
    )
    parser.add_argument(
        "figure_number",
        type=int,
        choices=[e.value for e in FigureMatcher],
        help="Number of the figure to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/",
        help="Output directory for the figure.",
    )

    args = parser.parse_args()
    plot.set_style()
    label = FigureMatcher(args.figure_number).name.lower()
    # data_interface = libdata.Loader.from_label(label)
    plotter = plot.AbstractPlotter.from_label(label)
    # plotter = plot.Plotter.from
    # if label == "simple_example":
    #     # data_interface = libdata.SimpleExampleLoader.from_raw_data()
    #     plotter = plot.SimpleExamplePlotter(data=data_interface)
    output_dir = pathlib.Path(args.output)
    ensure_output_path_exists(output_dir)
    plotter.make_and_write(output_dir)


if __name__ == "__main__":
    main()
