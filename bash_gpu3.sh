#!/bin/sh
#BSUB -q gpuv100
#BSUB -J testjob
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 18:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/errors/gpu_%J.err

nvidia-smi
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source $HOME/miniconda3/bin/activate
source venv_1/bin/activate

python3 main.py Deep_google 128 1e-6 1 600 2_1_sgd_wd1 0 SGD
python3 main.py Deep_google 128 1e-6 1 600 3_4_sgd_wd1 1 SGD
python3 main.py Deep_google 128 1e-6 1 600 6_7_sgd_wd1 3 SGD
python3 main.py Deep_google 128 1e-6 1 600 4_5_sgd_wd1 5 SGD