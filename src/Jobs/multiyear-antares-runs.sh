###!/bin/sh
### General options
### -- specify queue --
#BSUB -q man
### -- set the job Name --
#BSUB -J antares_multiyear_runs
### -- ask for number of cores (default: 1) --
#BSUB -n 35
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need X GB of memory per core/slot --
#BSUB -R "rusage[mem=1GB]"
### -- specify that we want the job to get killed if it exceeds X GB per core/slot --
#BSUB -M 1.1GB
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
#BSUB -o ./Logs/antares_multiyear_runs_%J.out
#BSUB -e ./Logs/antares_multiyear_runs_%J.err
# here follow the commands you want to execute with input.in as the input file

### Load modules and find binaries
module load R/4.2.3-mkl2023update2

### Get paths to binaries and Python-API for GAMS
export PATH=/zhome/c0/2/105719/Desktop/Antares-8.7.0/bin:$PATH
export PATH=/appl/gams/47.6.0:$PATH
export PATH=~/.pixi/bin:$PATH

for weather_year in 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016; do

  # Change weather year in config
  sed -i "s/^balmorel_weather_year:.*$/balmorel_weather_year: $weather_year/" Config.ini

  for name in noh noh2 h2 h2_lss h2_lss_h2t; do
    # Scenario name
    scenario_name="${name}_dispatch_WY${weather_year}"

    # Change scenario name in config
    sed -i "s/^SC:.*$/SC: $scenario_name/" Config.ini
    sed -i "s/^SC_folder:.*$/SC_folder: $name/" Config.ini

    # Run preprocessing
    # pixi run preprocessing -F --rerun-incomplete
    pixi run initialisation

    # Rename Config_SCX.ini to Config.ini (make active)
    # mv Config_${name}.ini ""

    # Running Master
    # ~/.pixi/bin/pixi run python Master.py

    # Running Balmorel
    # cd Balmorel
    # cp -f "simex_${name}"/* simex/
    #
    # cd "${name}/model"
    # mv balopt.opt balopt_invest.opt
    # mv balopt_dispatch.opt balopt.opt
    # gams Balmorel --scenario_name "${name}_dispatch_WY${weather_year}_Iter0" threads $LSB_DJOB_NUMPROC
    # mv balopt.opt balopt_dispatch.opt
    # mv balopt_invest.opt balopt.opt
    # cd ../../../

    for cluster_size in 1344; do
      for year in 2050; do
        # Running Peri-Processing
        pixi run periprocess $scenario_name $year $cluster_size

        if [ $? -ne 0 ]; then
          exit 1
        fi

        # Running Antares
        antares-8.7-solver Antares -n "${name}_WY${weather_year}_cl${cluster_size}_Iter0_Y-${year}" --parallel
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
