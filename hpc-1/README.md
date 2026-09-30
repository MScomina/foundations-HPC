## Deployment for Exercise 1

All steps are executed as SLURM batch jobs:

1. **Compile OSU**
	```bash
	sbatch compile_osu.sh
	```
	The script loads the required modules, downloads the OSU micro‑benchmarks, configures the build with `./configure` and then compiles and installs it with `make`.

2. **Run OSU**
	```bash
	sbatch run_osu.sh
	```
	The job executes the OSU benchmarks and writes a JSON file with the benchmark results.

3. **Plot results**
	```bash
	sbatch plot_job.sh
	```
	This job generates the graphs from the JSON data and stores them in the `results_plots` directory.

### Summary
After the three jobs finish you will find the plotted figures in `results_plots/` and the raw JSON data in the directory created by `run_osu.sh`.
---
   *Remember to use the proper formatting depending on what SLURM version and settings you are dealing with.*
   *ORFEO requires you to provide the account type when running the jobs: `sbatch -A <account-type> [command]`*