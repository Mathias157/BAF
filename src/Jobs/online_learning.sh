###!/bin/sh
### General options
### -- specify queue --
#BSUB -q man
### -- set the job Name --
#BSUB -J testing_online_learning
### -- ask for number of cores (default: 1) --
#BSUB -n 5
### -- specify that we need a certain architecture --
#BSUB -R "select[model == XeonGold6226R]"
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need X GB of memory per core/slot --
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds X GB per core/slot --
#BSUB -M 2.1GB
### -- set walltime limit: hh:mm --
#BSUB -W 10:00
### -- set the email address --
#BSUB -u ahego@dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./Logs/testing_online_learning_%J.out
#BSUB -e ./Logs/testing_online_learning_%J.err
# here follow the commands you want to execute with input.in as the input file

export PATH=/appl/gams/47.6.0:$PATH
export PATH=~/.pixi/bin:$PATH

### --- Run ---
pixi run python Workflow/Functions/online_learning.py \
  --scenario-name DO_D1W1 \
  --scenario-folder operun \
  --pretrain-epochs 50 \
  --update-epochs 200 \
  --days 1 \
  --n-scenarios 2 \
  --latent-dim 64 \
  --seed 0 \
  --batch-size 256 \
  --learning-rate 0.0005 \