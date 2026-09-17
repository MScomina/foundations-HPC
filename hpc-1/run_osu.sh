#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --job-name=osu_bcast_reduce
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
PATH=$PATH:${OSU_PATH}/libexec/osu-micro-benchmarks/mpi/collective
python3 tests.py