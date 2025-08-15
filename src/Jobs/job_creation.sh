###!/bin/sh
### General options
### -- specify queue --
#BSUB -q man
### -- set the job Name --
#BSUB -J Scenario
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
#BSUB -o ./Logs/Scenario_%J.out
#BSUB -e ./Logs/Scenario_%J.err
# here follow the commands you want to execute with input.in as the input file

### Load modules and find binaries
module load R/4.2.3-mkl2023update2

### Get paths to binaries and Python-API for GAMS
export PATH=/zhome/c0/2/105719/Desktop/Antares-8.7.0/bin:$PATH
export PATH=/appl/gams/47.6.0:$PATH

for name in Scenario; do
    # Rename Config_SCX.ini to Config.ini (make active)
    # mv Config_${name}.ini "Config.ini"

    # Running Master
    ~/.pixi/bin/pixi run python Master.py

    # Running Balmorel 
    #cd Balmorel/Scenario/model 
    #gams Balmorel     --scenario_name $name --threads $LSB_DJOB_NUMPROC --CAPCRED yes   --USEANTARESDATA yes  --AGGR1 yes --AGGR2 yes  --ADJUSTHYDRO yes

    # for year in 2040 2050; do
    #     # Running Peri-Processing
    #     python3 -m runpy "Workflow.Peri-Processing" $year $name

    #     # Running Antares
    # 	antares-8.6-solver Antares -n "${name}_Iter0_Y-${year}" --parallel
    # done

    # Running ConvergenceCriterion
    # python3 -m runpy "Workflow.ConvergenceCriterion" $name

    # Running Post-Processing
    # python3 -m runpy "Workflow.Post-Processing" $name
    # unzip Workflow/OverallResults/20240523-1035_LTFictDemFunc3MaxFlexDem_Results.zip

    
    # Running Analysis
    # python3 -m runpy "Workflow.Analysis" $name 

    # Rename Config.ini to Config_SCX.ini (make inactive)
    # mv Config.ini "Config_${name}.ini"
done