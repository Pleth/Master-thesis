#!/bin/sh
#BSUB -q gpuv100
#BSUB -J testjob
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/errors/gpu_%J.err

nvidia-smi
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source $HOME/miniconda3/bin/activate
source venv_1/bin/activate

#python3 main.py Deep_google 128 1e-6 15 800 2_4_test4 7
#python3 main.py Deep_google 128 1e-6 20 800 shuffle_test5 99
python3 main.py CNN_simple 128 1e-6 0.1 200 shuffle_wd01 99
python3 main.py CNN_simple 128 1e-6 1 200 shuffle_wd1 99
python3 main.py CNN_simple 128 1e-6 10 200 shuffle_wd10 99
python3 main.py CNN_simple 128 1e-6 20 200 shuffle_wd20 99
python3 main.py CNN_simple 128 1e-6 30 200 shuffle_wd30 99
python3 main.py CNN_simple 128 1e-6 40 200 shuffle_wd40 99
