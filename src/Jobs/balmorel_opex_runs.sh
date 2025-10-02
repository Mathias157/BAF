###!/bin/sh
### General options
### -- specify queue --
#BSUB -q man
### -- set the job Name --
#BSUB -J balmorel_operuns_and_antares
### -- ask for number of cores (default: 1) --
#BSUB -n 35
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need X GB of memory per core/slot --
#BSUB -R "rusage[mem=3GB]"
### -- specify that we want the job to get killed if it exceeds X GB per core/slot --
#BSUB -M 3.1GB
### -- set walltime limit: hh:mm --
#BSUB -W 96:00
### -- set the email address --
#BSUB -u mberos@dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./Logs/balmorel_operuns_and_antares_%J.out
#BSUB -e ./Logs/balmorel_operuns_and_antares_%J.err
# here follow the commands you want to execute with input.in as the input file

### Load modules and find binaries
module load R/4.2.3-mkl2023update2

### Get paths to binaries and Python-API for GAMS
export PATH=/zhome/c0/2/105719/Desktop/Antares-8.7.0/bin:$PATH
export PATH=/appl/gams/47.6.0:$PATH
export PATH=~/.pixi/bin:$PATH

for weather_year in 2000; do
  # Change weather year
  # sed -i "s/^balmorel_weather_year:.*$/balmorel_weather_year: $weather_year/" Config.ini

  # Run preprocessing
  # pixi run preprocessing -F --rerun-incomplete

  for name in noh noh2 h2 h2_lss h2_lss_h2t; do
    # Rename Config_SCX.ini to Config.ini (make active)
    # mv Config_${name}.ini ""
    scenario_name="${name}_eu_operun"

    if [ $name = "noh" ]; then
      h2_clustering_technique="vre_availability"
      clustering_name="h2vrehexo"
    else
      h2_clustering_technique="exogenous_demand"
      clustering_name="h2exohexo"
    fi

    # Running Master
    # ~/.pixi/bin/pixi run python Master.py

    # Running Balmorel
    cd Balmorel
    cp -f "simex_${name}"/* simex/

    cd "${name}/model"
    mv balopt.opt balopt_invest.opt
    mv balopt_dispatch.opt balopt.opt

    gams Balmorel --scenario_name "${scenario_name}_Iter0" threads $LSB_DJOB_NUMPROC

    if [ $? -ne 0 ]; then
      mv balopt.opt balopt_dispatch.opt
      mv balopt_invest.opt balopt.opt
      exit 1
    fi

    mv balopt.opt balopt_dispatch.opt
    mv balopt_invest.opt balopt.opt
    cd ../../../

    for cluster_size in 672; do
      for year in 2050; do
        # Running Peri-Processing
        pixi run periprocess $scenario_name $year $cluster_size --hydrogen-parameter-choice $h2_clustering_technique

        if [ $? -ne 0 ]; then
          exit 1
        fi

        # Running Antares
        antares-8.7-solver Antares -n "${name}_wy${weather_year}_${cluster_size}_${clustering_name}_Iter0_Y-${year}" --parallel
      done
    done

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
done
