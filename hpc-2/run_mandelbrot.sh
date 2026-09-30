#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --job-name=mandelbrot_execution
#SBATCH --cpus-per-task=16
#SBATCH --mem=15gb
#SBATCH --time=00:15:00
#SBATCH --output=./logs/mandelbrot_run%j.out

if [ ! -f .env ]; then
	cp .env.example .env
	echo "Created .env from .env.example."
fi

set -a; source .env set +a

module load "$MPI_MODULE"

MPIRUN_OPTS="-np 1 --bind-to none"
# Reminder: program format is ./2c pixel_x pixel_y real_btmleft imag_btmleft real_topright imag_topright i_max.
./2c 2048 2048 0.35787121 0.1081397025 0.35787122 0.1081397125 3000