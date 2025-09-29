#!/bin/bash

# specify job name

#BSUB -J profile_lines
#BSUB -o profile_lines_%J.out
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

python ./profile_lines.py