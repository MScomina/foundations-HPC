import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

# Note: part of the plotting code has been created with the help of AI, more precisely the model gpt-oss:20b.

# Location of the compiled binary
BIN = Path("./2c")

# Number of runs to average over for each configuration
N_RUNS = 5

# Region of the complex plane to explore (x_l, y_l, x_r, y_r)
REGION = (0.35787121, 0.1081397025, 0.35787122, 0.1081397125)

# Iteration limit for the Mandelbrot calculation
I_MAX = 3000

# Fixed problem size for strong‑scaling tests
STRONG_TOTAL_PIXELS = 2048 * 2048

# Starting tile size for weak‑scaling tests (per process/thread)
WEAK_PIXELS_PER_PROC = 512 * 512

# Numbers of MPI processes and OpenMP threads to test
MAX_CORES = 24
NUM_PROCESSES = [k for k in range(2, MAX_CORES+1, 2)]
NUM_THREADS = [k for k in range(2, MAX_CORES+1, 2)]

# Binding option for mpirun (e.g., "socket", "core", or "none")
BIND_TO = "none"

@dataclass
class Result:
	label: str
	value: float
	# Mean execution time for the configuration
	mean: float
	# Standard deviation of the execution times
	std: float


def run_command(cmd: list[str], env: dict | None = None) -> float:
	"""Execute *cmd* and return the elapsed wall‑time.

	The function uses :func:`subprocess.run` with ``check=True`` to make
	sure any non‑zero exit status is raised as an exception.
	"""

	start = time.perf_counter()
	# Preserve the existing environment unless a custom env is supplied.
	if env is not None:
		run_env = {**os.environ, **env}
	else:
		run_env = os.environ
	subprocess.run(cmd, env=run_env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	end = time.perf_counter()
	return end - start


def mpi_weak_scaling() -> list[Result]:
	"""Weak scaling for the MPI implementation.

	The number of MPI processes increases while the workload per process
	stays constant.
	"""

	results: list[Result] = []
	for p in NUM_PROCESSES:
		n_x = n_y = int((WEAK_PIXELS_PER_PROC * (p-1)) ** 0.5)	# p-1 because the master shouldn't count in the scaling.
		cmd = ["mpirun", "-np", str(p), "--bind-to", BIND_TO, str(BIN), str(n_x), str(n_y), *map(str, REGION), str(I_MAX)]
		# OpenMP threads forced to 1
		env = {"OMP_NUM_THREADS": "1"}
		times = [run_command(cmd, env) for _ in range(N_RUNS)]
		avg = sum(times) / len(times)
		std = math.sqrt(sum((t - avg) ** 2 for t in times) / len(times))
		results.append(Result(f"MPI weak {p}p", p, avg, std))
	return results


def mpi_strong_scaling() -> list[Result]:
	"""Strong scaling for the MPI implementation.

	The total workload stays constant while the number of processes
	increases.
	"""

	results: list[Result] = []
	for p in NUM_PROCESSES:
		n_x = n_y = int((STRONG_TOTAL_PIXELS) ** 0.5)
		cmd = ["mpirun", "-np", str(p), "--bind-to", BIND_TO, str(BIN), str(n_x), str(n_y), *map(str, REGION), str(I_MAX)]
		env = {"OMP_NUM_THREADS": "1"}
		times = [run_command(cmd, env) for _ in range(N_RUNS)]
		avg = sum(times) / len(times)
		std = math.sqrt(sum((t - avg) ** 2 for t in times) / len(times))
		results.append(Result(f"MPI strong {p}p", p, avg, std))
	return results


def openmp_weak_scaling() -> list[Result]:
	"""Weak scaling for the OpenMP implementation.

	The number of OpenMP threads increases while the workload per thread
	stays constant.  The binary is executed with a single MPI process.
	"""

	results: list[Result] = []
	for t in NUM_THREADS:
		n_x = n_y = int((WEAK_PIXELS_PER_PROC * t) ** 0.5)
		cmd = ["mpirun", "-np", "1", "--bind-to", BIND_TO, str(BIN), str(n_x), str(n_y), *map(str, REGION), str(I_MAX)]
		env = {"OMP_NUM_THREADS": str(t)}
		times = [run_command(cmd, env) for _ in range(N_RUNS)]
		avg = sum(times) / len(times)
		std = math.sqrt(sum((t - avg) ** 2 for t in times) / len(times))
		results.append(Result(f"OpenMP weak {t}t", t, avg, std))
	return results


def openmp_strong_scaling() -> list[Result]:
	"""Strong scaling for the OpenMP implementation.

	The total workload stays constant while the number of threads
	increases.
	"""

	results: list[Result] = []
	for t in NUM_THREADS:
		n_x = n_y = int((STRONG_TOTAL_PIXELS) ** 0.5)
		cmd = ["mpirun", "-np", "1", "--bind-to", BIND_TO, str(BIN), str(n_x), str(n_y), *map(str, REGION), str(I_MAX)]
		env = {"OMP_NUM_THREADS": str(t)}
		times = [run_command(cmd, env) for _ in range(N_RUNS)]
		avg = sum(times) / len(times)
		std = math.sqrt(sum((t - avg) ** 2 for t in times) / len(times))
		results.append(Result(f"OpenMP strong {t}t", t, avg, std))
	return results


def plot_results(results: list[Result], title: str, filename: str) -> None:
	"""Plot a single benchmark series.

	The x‑axis shows the number of processes/threads and the y‑axis shows
	the averaged execution time. Error bars represent the standard
	deviation calculated for each configuration.
	"""

	values = [r.value for r in results]
	times = [r.mean for r in results]
	stds = [r.std for r in results]

	plt.figure(figsize=(6, 4))
	# Plot mean line with markers
	plt.plot(values, times, marker="o", linestyle="-", label="Mean")
	plt.xlabel("# of processes / threads")
	plt.ylabel("Wall‑time (s)")
	plt.title(title)
	plt.grid(True)
	plt.tight_layout()
	# Shade area between mean ± std
	lower = [m - s for m, s in zip(times, stds)]
	upper = [m + s for m, s in zip(times, stds)]
	plt.fill_between(values, lower, upper, color="gray", alpha=0.3, label="Std Dev")
	plt.legend()
	plt.savefig(filename)
	plt.close()


def main() -> None:
    out_dir = Path("./benchmark_results")
    out_dir.mkdir(exist_ok=True)

    def run_and_plot(func: callable, title: str, pdf_name: str) -> None:
        pdf_path = out_dir / pdf_name
        if pdf_path.exists():
            print(f"{pdf_name} already exists – skipping benchmark.")
            return
        print(f"Running {title}...")
        results = func()
        print(f"{title} done.")
        plot_results(results, title, pdf_path)

    run_and_plot(mpi_weak_scaling, "MPI Weak Scaling", "mpi_weak.pdf")
    run_and_plot(mpi_strong_scaling, "MPI Strong Scaling", "mpi_strong.pdf")
    run_and_plot(openmp_weak_scaling, "OpenMP Weak Scaling", "omp_weak.pdf")
    run_and_plot(openmp_strong_scaling, "OpenMP Strong Scaling", "omp_strong.pdf")
    print("All plots saved to", out_dir, ".")

if __name__ == "__main__":
	main()