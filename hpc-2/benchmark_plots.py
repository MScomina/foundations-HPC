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
N_RUNS = 10

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
		n_x = n_y = int((WEAK_PIXELS_PER_PROC * p) ** 0.5)
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
	print("Running MPI weak scaling…")
	mpi_weak = mpi_weak_scaling()
	print("MPI weak done.")
	print("Running MPI strong scaling…")
	mpi_strong = mpi_strong_scaling()
	print("MPI strong done.")
	print("Running OpenMP weak scaling…")
	omp_weak = openmp_weak_scaling()
	print("OpenMP weak done.")
	print("Running OpenMP strong scaling…")
	omp_strong = openmp_strong_scaling()
	print("OpenMP strong done.")

	out_dir = Path("./benchmark_results")
	out_dir.mkdir(exist_ok=True)
	plot_results(mpi_weak, "MPI Weak Scaling", out_dir / "mpi_weak.png")
	plot_results(mpi_strong, "MPI Strong Scaling", out_dir / "mpi_strong.png")
	plot_results(omp_weak, "OpenMP Weak Scaling", out_dir / "omp_weak.png")
	plot_results(omp_strong, "OpenMP Strong Scaling", out_dir / "omp_strong.png")
	print("All plots saved to", out_dir, ".")


if __name__ == "__main__":
	main()