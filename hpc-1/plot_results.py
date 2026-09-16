import argparse
import json
import pathlib
import statistics

import matplotlib.pyplot as plt


def _average_runs(runs: list[list[list]]) -> tuple[list[float], list[float], list[float]]:
    if not runs:
        return [], [], []

    n_rows = len(runs[0]) - 1

    sizes: list[float] = []
    means: list[float] = []
    stds: list[float] = []

    for row_idx in range(n_rows):
        values: list[float] = []
        for run in runs:
            data_row = run[row_idx + 1]
            try:
                size = float(data_row[0])
            except ValueError:
                size = 0.0
            values.append(float(data_row[1]))
            if not sizes:
                sizes.append(size)

        means.append(statistics.mean(values))
        if len(values) > 1:
            stds.append(statistics.stdev(values))
        else:
            stds.append(0.0)

    return sizes, means, stds


def plot_task(task_name: str, task_data: dict, out_dir: pathlib.Path):
    plt.figure(figsize=(10, 6))
    for alg, np_map in task_data.items():
        for np_count, runs in np_map.items():
            sizes, means, stds = _average_runs(runs)
            if not sizes:
                continue
            label = f"alg {alg}, {np_count}p"
            plt.plot(sizes, means, marker="o", label=label)
            # Confidence interval (95%) assuming normal distribution
            if len(runs) > 1 and any(stds):
                # z value for 95% CI
                z = 1.96
                ci_upper = [m + z * s / (len(runs) ** 0.5) for m, s in zip(means, stds)]
                ci_lower = [m - z * s / (len(runs) ** 0.5) for m, s in zip(means, stds)]
                plt.fill_between(sizes, ci_lower, ci_upper, alpha=0.2)

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