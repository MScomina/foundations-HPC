"""Plot OSU benchmark results produced by :mod:`tests.py`.

The :mod:`tests.py` script writes a JSON file named ``the_results_of_osu.json``
containing a nested dictionary structure::

    {
        "osu_bcast": {
            "1": { 2: [run1, run2, ...], 4: [...], ... },
            "2": { ... }
        },
        "osu_gather": { ... },
        "naive model": {
            "1": { 2: [run1, run2, ...] }
        }
    }

Each *run* is a list of rows read from the OSU benchmark output.  The
first row is a header containing the column names; all subsequent rows are
numeric values.

This script reads the JSON file, averages the numeric values over all
repetitions for a given ``algorithm``/``num_processes`` combination, and
produces a matplotlib figure for each benchmark task.  The resulting PNG
files are written to a directory supplied via the ``--out`` command line
option (default: ``results_plots``).
"""

import argparse
import json
import pathlib
import statistics

import matplotlib.pyplot as plt


def _average_runs(runs: list[list[list]]) -> tuple[list[float], list[float]]:
    """Return (sizes, values) averaged over ``runs``.

    Each element of ``runs`` is a list of rows (as produced by
    :func:`tests.parse_osu_output`).  The first row is the header.
    The function assumes that all runs have the same header and the same
    set of sizes in the same order.
    """
    if not runs:
        return [], []

    # The header row is not needed for the averaging logic
    n_rows = len(runs[0]) - 1  # number of data rows

    sizes: list[float] = []
    avg_values: list[float] = []

    for row_idx in range(n_rows):
        # Collect the values for this row from all runs
        values = []
        for run in runs:
            data_row = run[row_idx + 1]  # skip header
            # Some benchmarks use a string for size; cast to float
            try:
                size = float(data_row[0])
            except ValueError:
                size = 0.0
            values.append(float(data_row[1]))
            if not sizes:
                sizes.append(size)

        avg_values.append(statistics.mean(values))

    return sizes, avg_values


def plot_task(task_name: str, task_data: dict, out_dir: pathlib.Path):
    """Create a figure for a single benchmark task.

    Parameters
    ----------
    task_name:
        Name of the task (e.g., ``osu_bcast``).
    task_data:
        Nested dictionary mapping algorithm → num_processes → list of runs.
    out_dir:
        Directory where the PNG file will be written.
    """
    plt.figure(figsize=(10, 6))
    for alg, np_map in task_data.items():
        for np_count, runs in np_map.items():
            sizes, values = _average_runs(runs)
            if not sizes:
                continue
            label = f"alg {alg}, {np_count}p"
            plt.plot(sizes, values, marker="o", label=label)

    plt.title(f"{task_name} results")
    plt.xlabel("Size")
    plt.ylabel("Bandwidth [MB/s]" if task_name != "naive model" else "Latency [us]")
    plt.grid(True)
    plt.legend()
    out_file = out_dir / f"{task_name}.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OSU benchmark results.")
    parser.add_argument(
        "--json",
        default="the_results_of_osu.json",
        help="Path to the JSON file produced by tests.py (default: current dir)",
    )
    parser.add_argument(
        "--out",
        default="results_plots",
        help="Output directory for PNG files",
    )
    args = parser.parse_args()

    json_path = pathlib.Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    out_dir = pathlib.Path(args.out)
    for task_name, task_data in data.items():
        plot_task(task_name, task_data, out_dir)
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()