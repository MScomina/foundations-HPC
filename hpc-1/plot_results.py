import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def plot_task(task_name: str, task_data: dict, out_dir: pathlib.Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_cores = {2, 4, 8, 16, 32, 48}

    variants = list(task_data.keys())
    n_variants = len(variants)
    fig, axes = plt.subplots(nrows=n_variants, ncols=1, figsize=(8, 4 * n_variants), sharex=False)
    if n_variants == 1:
        axes = [axes]
    for idx, variant in enumerate(variants):
        variant_data = task_data[variant]
        ax = axes[idx]
        for core_str, runs in variant_data.items():
            try:
                core = int(core_str)
            except ValueError:
                continue
            if core not in allowed_cores:
                continue
            from collections import defaultdict
            latency_by_size = defaultdict(list)
            for run in runs:
                data_rows = run[1:] if isinstance(run, list) and len(run) > 0 else run
                for pair in data_rows:
                    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                        continue
                    size, latency = pair
                    latency_by_size[size].append(latency)
            if not latency_by_size:
                continue
            sizes = sorted(latency_by_size.keys())
            means = []
            cis = []  # 95% confidence interval (mean ± 1.96*std/sqrt(n))
            for size in sizes:
                values = latency_by_size[size]
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                n = len(values)
                ci = 1.96 * std / np.sqrt(n) if n > 1 else 0
                means.append(mean)
                cis.append(ci)
            ax.plot(sizes, means, label=f"{core} cores")
            lower = [m - c for m, c in zip(means, cis)]
            upper = [m + c for m, c in zip(means, cis)]
            ax.fill_between(sizes, lower, upper, alpha=0.2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Packet Size (Bytes)")
        ax.set_ylabel("Avg Latency (us)")
        ax.set_title(f"{variant}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.3)
        ax.legend()

    fig.suptitle(f"{task_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_dir / f"{task_name}.png"
    fig.savefig(out_path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot OSU benchmark results.")
    parser.add_argument("--json", default="osu_bcast_reduce_results.json",
                        help="Path to the JSON file produced by tests.py")
    parser.add_argument("--out", default="results_plots",
                        help="Output directory for PNG files")
    args = parser.parse_args()

    json_path = pathlib.Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))

    out_dir = pathlib.Path(args.out)
    for task_name, task_data in data.items():
        plot_task(task_name, task_data, out_dir)
    print(f"Plots written to {out_dir}.")


if __name__ == "__main__":
    main()