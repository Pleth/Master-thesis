#!/bin/sh
#BSUB -q gpuv100
#BSUB -J testjob
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
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
python3 main.py CNN_simple 128 1e-8 0.1 100 shuffle1_wd01 99 SGD
python3 main.py CNN_simple 128 1e-8 1 100 shuffle1_wd1 99 SGD
python3 main.py CNN_simple 128 1e-8 10 100 shuffle1_wd10 99 SGD
python3 main.py CNN_simple 128 1e-8 0.1 100 2_4_wd01 7 SGD
python3 main.py CNN_simple 128 1e-8 1 100 2_4_wd1 7 SGD
python3 main.py CNN_simple 128 1e-8 10 100 2_4_wd10 7 SGD
