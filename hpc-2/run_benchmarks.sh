#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --job-name=mandelbrot_benchmark
#SBATCH --ntasks-per-node=24
#SBATCH --ntasks=48
#SBATCH --exclusive
#SBATCH --mem=15gb
#SBATCH --time=01:45:00
#SBATCH --output=./logs/output%j.out

if [ ! -f .env ]; then
	cp .env.example .env
	echo "Created .env from .env.example."
fi

set -a; source .env set +a

module load "$MPI_MODULE"

python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate

pip install -r ../requirements.txt
python3 benchmark_plots.py