import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

VARIANT_TITLES = {
    "naive_model": {
        1: "p2p"
    },
    "osu_bcast": {
        0: "ignore",
        1: "basic linear",
        2: "chain",
        3: "pipeline",
        4: "split binary tree",
        5: "binary tree",
        6: "binomial tree"
    },
    "osu_reduce": {
        0: "ignore", 
        1: "linear", 
        2: "chain",
        3: "pipeline", 
        4: "binary", 
        5: "binomial",
        6: "in-order_binary", 
        7: "rabenseifner"
    },
}

FIXED_SIZES = [512, 65536, 1048576]


def aggregate_latency_per_n_proc(task_data: dict) -> dict[int, list[float]]:
    """
        Flatten all latency values per number of processes from a dict n_proc->runs.
    """
    proc_latency: dict[int, list[float]] = defaultdict(list)
    for proc_str, runs in task_data.items():
        try:
            proc = int(proc_str)
        except ValueError:
            continue
        latency_by_size = aggregate_latency_by_size(runs)
        for vals in latency_by_size.values():
            proc_latency[proc].extend(vals)
    return proc_latency

    
def plot_task(task_name: str, task_data: dict, out_dir: pathlib.Path) -> None:
    """
        Create one figure for a task and save it to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed_n_proc = {2, 4, 8, 16, 32, 48}
    variants = list(task_data.keys())
    n_variants = len(variants)
    fig, axes = plt.subplots(
        nrows=n_variants, ncols=1, figsize=(8, 4 * n_variants), sharex=False
    )
    if n_variants == 1:
        axes = [axes]
    for idx, variant in enumerate(variants):
        plot_variant(
            ax=axes[idx],
            variant_name=variant,
            variant_data=task_data[variant],
            allowed_n_proc=allowed_n_proc,
            task_name=task_name,
        )
    fig.suptitle(f"{task_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_dir / f"{task_name}.png"
    fig.savefig(out_path)
    plt.close(fig)


def compute_stats(values: list[float]):
    """
        Return mean and 95% CI for a list of values.
    """
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0
    return mean, ci


def plot_normalized_latency(data: dict[str, dict], out_dir: pathlib.Path) -> None:
    """
        Single figure: one subplot per algorithm, normalized latency vs number of processes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    algos = [a for a in data if a != "naive_model"]
    n = len(algos)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for idx, algo in enumerate(algos):
        for variant_name, variant_data in data[algo].items():
            proc_lat = aggregate_latency_per_n_proc(variant_data)
            n_processes = sorted(proc_lat)
            means, _ = zip(
                *[compute_stats(proc_lat[c]) for c in n_processes]
            )
            norm = means[0]
            norm_means = [m / norm for m in means]
            axes[idx].plot(n_processes, norm_means, label=VARIANT_TITLES[algo][int(variant_name)])
        axes[idx].set_title(algo)
        axes[idx].set_xlabel("N. Processes")
        axes[idx].set_ylabel("Norm. Avg Latency")
        axes[idx].grid(True, which="both", linestyle="--", linewidth=0.3)
        axes[idx].legend(title="Variant")
    fig.suptitle("Normalized Avg Latency per Algorithm")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_dir / "normalized_latency.png"
    fig.savefig(out_path)
    plt.close(fig)


def plot_variant(
    *,
    ax,
    variant_name: str,
    variant_data: dict,
    allowed_n_proc: set[int],
    task_name: str,
):
    """
        Plot a single variant on *ax*.
    """
    for n_proc_str, runs in variant_data.items():
        try:
            n_proc = int(n_proc_str)
        except ValueError:
            continue
        if n_proc not in allowed_n_proc:
            continue
        latency_by_size = aggregate_latency_by_size(runs)
        if not latency_by_size:
            continue
        sizes, means, cis = compute_means_and_cis(latency_by_size)
        plot_with_ci_and_data(ax, sizes, means, cis, n_proc=n_proc)
    configure_axes(ax, task_name, variant_name)


def aggregate_latency_by_size(runs: list) -> dict[int, list[float]]:
    """
        Collect latency values grouped by packet size.
    """
    latency_by_size = defaultdict(list)
    for run in runs:
        data_rows = run[1:] if isinstance(run, list) and len(run) > 0 else run
        for pair in data_rows:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            size, latency = pair
            latency_by_size[size].append(latency)
    return latency_by_size


def compute_means_and_cis(latency_by_size: dict[int, list[float]]) -> tuple[list[int], list[float], list[float]]:
    """
        Return sorted sizes, mean latencies and 95% CI arrays.
    """
    sizes = sorted(latency_by_size.keys())
    means = []
    cis = []
    for size in sizes:
        values = latency_by_size[size]
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)
        ci = 1.96 * std / np.sqrt(n) if n > 1 else 0
        means.append(mean)
        cis.append(ci)
    return sizes, means, cis


def plot_with_ci_and_data(
    ax,
    x,
    y,
    yerr,
    label=None,
    marker="o",
    n_proc=None,
):
    """
        Unified helper for plotting.
    """
    if n_proc is None:
        ax.plot(x, y, marker=marker, label=label)
    else:
        ax.plot(x, y, label=f"{n_proc}")
    lower = [max(yi - ei, 0) for yi, ei in zip(y, yerr)]
    upper = [yi + ei for yi, ei in zip(y, yerr)]
    ax.fill_between(x, lower, upper, alpha=0.2)


def configure_axes(ax, task_name: str, variant: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Packet Size (Bytes)")
    ax.set_ylabel("Avg Latency (us)")
    title_dict = VARIANT_TITLES[task_name]
    title = title_dict[int(variant)]
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.3)
    ax.legend(title="N. Processes")


def plot_fixed_sizes(task_name: str, task_data: dict, out_dir: pathlib.Path) -> None:
    """
        Plot latency vs. number of processes for a *single* packet size.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    n_sizes = len(FIXED_SIZES)
    fig, axes = plt.subplots(n_sizes, 1, figsize=(8, 4 * n_sizes), sharex=False)
    if n_sizes == 1:
        axes = [axes]
    for ax, size in zip(axes, FIXED_SIZES):
        for variant_name, variant_data in task_data.items():
            n_proc_means = {}
            n_proc_cis = {}
            for n_proc_str, runs in variant_data.items():
                try:
                    n_proc = int(n_proc_str)
                except ValueError:
                    continue
                latency_by_size = aggregate_latency_by_size(runs)
                if size not in latency_by_size:
                    continue
                vals = latency_by_size[size]
                mean, ci = compute_stats(vals)
                n_proc_means[n_proc] = mean
                n_proc_cis[n_proc] = ci
            if not n_proc_means:
                continue
            n_procs = sorted(n_proc_means)
            means = [n_proc_means[c] for c in n_procs]
            cis = [n_proc_cis[c] for c in n_procs]
            label = VARIANT_TITLES[task_name][int(variant_name)]
            plot_with_ci_and_data(ax, n_procs, means, cis, label=label, marker=None)
            ax.set_xlabel("N. Processes")
            ax.set_ylabel("Avg Latency (us)")
            ax.set_title(f"{task_name} – fixed size {size} B")
            ax.grid(True, which="both", linestyle="--", linewidth=0.3)
            ax.legend(title="Variant")

    ax.set_xlabel("N. Processes")
    ax.set_ylabel("Avg Latency (us)")
    ax.set_title(f"{task_name} – fixed size {size} B")
    ax.grid(True, which="both", linestyle="--", linewidth=0.3)
    fig.tight_layout(pad=1.2)
    out_path = out_dir / f"{task_name}_fixed.png"
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
        if task_name in ("osu_bcast", "osu_reduce"):
            plot_fixed_sizes(task_name, task_data, out_dir)
    plot_normalized_latency(data, out_dir)
    print(f"Plots written to {out_dir}.")


if __name__ == "__main__":
    main()