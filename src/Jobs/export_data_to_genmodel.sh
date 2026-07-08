###!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J export_data_to_generative_model
### -- ask for number of cores (default: 1) --
#BSUB -n 10
### -- specify that we need a certain architecture --
#BSUB -R "select[model == XeonGold6226R]"
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need X GB of memory per core/slot --
#BSUB -R "rusage[mem=10GB]"
### -- specify that we want the job to get killed if it exceeds X GB per core/slot --
#BSUB -M 10.1GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
#BSUB -u mberos@dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./Logs/export_data_to_generative_model_%J.out
#BSUB -e ./Logs/export_data_to_generative_model_%J.err
# here follow the commands you want to execute with input.in as the input file

### Get paths to binaries and Python-API for GAMS
export PATH=/appl/gams/50.4.1:$PATH

pixi run python Workflow/Functions/export_data_to_generative_model.py base

