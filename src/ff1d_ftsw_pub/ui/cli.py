import argparse

from ..lib import plot


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for the publication."
    )
    parser.add_argument(
        "figure_number", type=int, help="Number of the figure to generate."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/",
        help="Output directory for the figure.",
    )

    args = parser.parse_args()
    plot.set_style()
    if args.figure_number == 4:
        plotter = plot.Fig04()
        plotter.save(args.output)


if __name__ == "__main__":
    main()
