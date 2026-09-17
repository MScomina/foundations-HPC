import argparse as ap
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path


def parse_osu_output(output: bytes) -> list[list[str | float]]:
    """
    Convert the raw stdout of an osu benchmark into a list-of-lists structure.

    The first line that starts with ``#`` contains the column names – it is
    stripped from the ``#`` and saved as the header.  All other lines are
    numeric results converted to floats.
    """
    text = output.decode("utf-8")
    rows: list[list[str | float]] = []

    for i, line in enumerate(text.splitlines()):
        if not line:
            continue

        # Header line (starts with '#')
        if line.startswith("#"):
            # Skip lines that are comments or extra headers
            if text.splitlines()[i + 1].startswith("#"):
                continue
            columns = [c.strip() for c in line.split("  ") if c not in ("#", "")]
            if columns:
                columns[0] = columns[0][2:]
                rows.append(columns)
            continue

        # Data line
        vals = [float(v.strip()) for v in line.split("  ") if v not in ("#", "")]
        rows.append(vals)

    return rows

def run_benchmark(
    cmd: list[str],
) -> list[list[str | float]]:
    """
    Helper that executes a command under mpirun and returns parsed output.
    """
    out, _ = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=os.environ).communicate()
    return parse_osu_output(out)

def main(osu_loc: str | None = None) -> None:
    if osu_loc is None:
        comp_path = Path(os.getenv("OSU_PATH", "."))
        osu_loc = (comp_path / "libexec/osu-micro-benchmarks/mpi/collective")
    else:
        osu_loc = Path(osu_loc)

    osu_path = Path(osu_loc)

    # Build MPI command
    base_cmd: list[str] = [
        "mpirun",
        "--map-by",
        "NODE",
        "--mca",
        "coll_tuned_use_dynamic_rules",
        "true",
        "--mca",
    ]

    algorithms = ["1", "3", "6"]
    tasks = (
        ("osu_bcast", "coll_tuned_bcast_algorithm"),
        ("osu_reduce", "coll_tuned_reduce_algorithm"),
    )

    n_iterations: int = int(os.environ["N_ITERATIONS"])
    out_file = "./osu_bcast_reduce_results.json"
    try:
        n_cpus_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    except KeyError as exc:
        raise RuntimeError(
            "SLURM_NTASKS_PER_NODE not found in environment "
            "(run this script inside a SLURM job or set the variable manually)."
        ) from exc

    results: defaultdict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Run MPI benchmark
    for task_name, alg_flag in tasks:
        print(f"Running task: {task_name}")
        for alg in algorithms:
            print(f"  Algorithm {alg}")
            for np_count in range(2, (n_cpus_per_node * 2) + 1, 2):
                print(f"    Processes: {np_count}")
                for _ in range(n_iterations):
                    cmd = (
                        base_cmd
                        + [alg_flag]
                        + [alg]
                        + ["-np", str(np_count)]
                        + [str(osu_path / task_name)]
                    )
                    results[task_name][alg][np_count].append(run_benchmark(cmd))

    lat_cmd_base = ["mpirun", "--map-by", "NODE"]
    lat_binary = (
        Path(os.getenv("OSU_PATH", "."))
        / "libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency"
    )
    for _ in range(n_iterations):
        cmd = lat_cmd_base + ["-np", "2", str(lat_binary)]
        data = run_benchmark(cmd)
        if data:
            data[0] = ["Size", "Latency (us)"]
        results["naive model"][1][2].append(data)


    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=4)


    print(f"Results written to {out_file}.")

if __name__ == "__main__":
    parser = ap.ArgumentParser(
        description="Run OSU MPI micro‑benchmarks and collect results."
    )
    parser.add_argument("--osu-loc", help="Path to the compiled OSU benchmark directory.")
    args = parser.parse_args()
    main(args.osu_loc)