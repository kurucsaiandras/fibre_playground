#!/bin/bash

# specify job name
NAME="fibre_sim_prop_biglr"

#BSUB -J fibre_sim_prop_biglr
#BSUB -o fibre_sim_prop_biglr_%J.out
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

python ./spring_system_3d.py --jobname $NAME