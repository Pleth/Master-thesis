#!/bin/sh
#BSUB -q hpc
#BSUB -J Tests
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 14:00
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/errors/Error_%J.err

source $HOME/miniconda3/bin/activate
source venv_1/bin/activate

python3 main.py 45km_test train
python3 main.py laser_test train
python3 main.py synth_test train
