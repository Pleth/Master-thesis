#!/bin/sh
#BSUB -q gpuv100
#BSUB -J testjob
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

nvidia-smi
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source $HOME/miniconda3/bin/activate
source venv_1/bin/activate

python3 main.py Deep_linear
python3 main.py Deep_google