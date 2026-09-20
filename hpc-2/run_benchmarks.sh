#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --job-name=mandelbrot_benchmark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=6
#SBATCH --ntasks=24
#SBATCH --exclusive
#SBATCH --mem=15gb
#SBATCH --time=00:45:00
#SBATCH --output=./logs/mandelbrot_out%j.out

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