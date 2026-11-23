#!/bin/bash

#SBATCH --job-name=TP2_Re
#SBATCH --partition=externos
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=output/output-%x-%j.txt
#SBATCH --error=errors/error-%x-%j.txt

mpiexec $1
