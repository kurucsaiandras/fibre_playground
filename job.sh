#!/bin/bash
#BSUB -J fibre_sim
#BSUB -o fibre_sim%J.out
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=2048]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -W 24:00

python ./spring_system_3d.py