#!/bin/bash

#SBATCH --job-name=TP1_Re                   # nombre para identificar el trabajo. Por defecto es el nombre del script

#SBATCH --partition=externos                # partición a la que se enviará el trabajo, si no se especifica por defecto irá a la partición “debug” (ver particiones)

#SBATCH --ntasks=2                          # cantidad de tareas(procesos)
#SBATCH --tasks-per-node=1                  # cantidad de procesos por nodo, para que distribuya entre varios nodos

#SBATCH --output=output/output-%x-%j.txt    # la salida y error estandar van a este archivo. Si no es especifca es slurm-%j.out (donde %j es el Job ID)
#SBATCH --error=errors/error-%x-%j.txt      # si se especifica, la salida de error va por separado a este archivo


# Aqui comienzan los comandos
mpiexec $1
