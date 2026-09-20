#!/bin/bash
#SBATCH --partition=THIN
#SBATCH --job-name=mandelbrot_compilation
#SBATCH --cpus-per-task=16
#SBATCH --mem=15gb
#SBATCH --time=00:10:00
#SBATCH --output=./logs/compilation%j.out
set -a; source .env set +a

module load "$MPI_MODULE"
module load "$CMAKE_MODULE"

cmake .
make