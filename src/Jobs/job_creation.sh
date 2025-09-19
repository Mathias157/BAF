###!/bin/sh
### General options
### -- specify queue --
#BSUB -q man
### -- set the job Name --
#BSUB -J new_runs_for_simex
### -- ask for number of cores (default: 1) --
#BSUB -n 10
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need X GB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds X GB per core/slot --
#BSUB -M 4.1GB
### -- set walltime limit: hh:mm --
#BSUB -W 168:00
### -- set the email address --
#BSUB -u mberos@dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./Logs/new_runs_for_simex_%J.out
#BSUB -e ./Logs/new_runs_for_simex_%J.err
# here follow the commands you want to execute with input.in as the input file

### Load modules and find binaries
module load R/4.2.3-mkl2023update2

### Get paths to binaries and Python-API for GAMS
export PATH=/zhome/c0/2/105719/Desktop/Antares-8.7.0/bin:$PATH
export PATH=/appl/gams/47.6.0:$PATH
export PATH=~/.pixi/bin:$PATH

for name in noh noh2 h2; do
  # Rename Config_SCX.ini to Config.ini (make active)
  # mv Config_${name}.ini "Config.ini"

  # Running Master
  # ~/.pixi/bin/pixi run python Master.py

  # Running Balmorel
  cd "Balmorel/${name}/model"
  gams Balmorel --scenario_name "${name}_fullyear2_Iter0" threads $LSB_DJOB_NUMPROC
  cd ../../

  cp -r simex "simex_${name}"
  cd ../

  # for cluster_size in 168; do
  #   for year in 2050; do
  #     # Running Peri-Processing
  #     pixi run periprocess $name $year $cluster_size
  #
  #     if [ $? -ne 0 ]; then
  #       exit 1
  #     fi
  #
  #     # Running Antares
  #     antares-8.7-solver Antares -n "${name}_h2surhheat_Iter0_Y-${year}" --parallel
  #   done
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
