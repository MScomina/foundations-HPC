## Using the Mandelbrot Benchmarks

All steps are executed as SLURM batch jobs. The workflow consists of three stages:

1. **Compile** – build the Mandelbrot program.
   ```bash
   sbatch compile.sh
   ```
   The script loads the required modules, creates the build directory, runs CMake, and then compiles the source code. The resulting binaries (`2c` and `mandelbrot`) are placed in the same directory as the script, i.e., the `hpc-2/` folder.

2. **Run the full benchmark** – execute the benchmarks across multiple nodes.
   ```bash
   sbatch run_benchmarks.sh
   ```
   This job launches the Python benchmark harness (`benchmark_plots.py`) which in turn drives the C++ program on the requested number of nodes.

3. **Run a single instance** – launch one copy of the 2c program.
   ```bash
   sbatch run_mandelbrot.sh
   ```
   The script simply submits a single‑node SLURM job that runs the compiled `2c` binary.

### Output
* Benchmark logs and plots are written to the `logs/` and `results_plots/` directories inside `hpc-2/`.
* The Mandelbrot computation produces an output image `mandelbrot.pgm` in the same folder.
* The `run_benchmarks.sh` script also produces a `benchmark_plots.py`‑generated report.