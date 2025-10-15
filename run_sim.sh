#!/bin/bash

# specify job name
NAME="fibre_sim"
CONFIG="default"

#BSUB -J fibre_sim
#BSUB -o fibre_sim_%J.out
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

python ./fibre_simulation.py --job_name $NAME --config $CONFIG