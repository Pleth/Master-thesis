#!/bin/sh
#BSUB -q gpuv100
#BSUB -J testjob
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 05:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/errors/gpu_%J.err

nvidia-smi
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source $HOME/miniconda3/bin/activate
source venv_1/bin/activate

python3 main.py Deep_google 128 1e-5 10 100 2_1 0
python3 main.py Deep_google 128 1e-5 10 100 3_4 1
python3 main.py Deep_google 128 1e-5 10 100 3_2 2
python3 main.py Deep_google 128 1e-5 10 100 6_7 3
python3 main.py Deep_google 128 1e-5 10 100 5_6 4
python3 main.py Deep_google 128 1e-5 10 100 1_7 5
python3 main.py Deep_google 128 1e-5 10 100 5_4 6
