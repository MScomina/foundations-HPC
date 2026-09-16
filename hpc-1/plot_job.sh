#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --job-name=plot_osu_results
#SBATCH --cpus-per-task=12
#SBATCH --mem=15gb
#SBATCH --time=00:10:00
#SBATCH --output=plot_%j.out
python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate

pip install matplotlib
python3 plot_results.py