#!/bin/bash
#SBATCH --job-name=python
#SBATCH --gres=gpu:1
#SBATCH --output=./modelM1/slurm7.out
#SBATCH --error=./modelM1/slurm.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
  
srun python mdebug_.py -n 2 -e 5000 -g 0  -x 1 -t /users/xudong.zhang/data/train.conll -d /users/xudong.zhang/data/dev.conll -u /users/xudong.zhang/data/test.conll -y ./data/glove.6B.100d.txt


#srun python mdebug.py -n 2 -e 5000 -g 0  -x 1 -t /users/xudong.zhang/cdata/Utrain.txt -d /users/xudong.zhang/cdata/Udev.txt -u /users/xudong.zhang/cdata/Utest.txt -y /users/xudong.zhang/cdata/cpretrained.txt
