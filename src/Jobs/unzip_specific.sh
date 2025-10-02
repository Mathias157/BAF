###!/bin/sh
### General options
### -- specify queue --
#BSUB -q man
### -- set the job Name --
#BSUB -J Unzip
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify that we need a certain architecture --
#BSUB -R "select[model == XeonGold6226R]"
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need X GB of memory per core/slot --
#BSUB -R "rusage[mem=1GB]"
### -- specify that we want the job to get killed if it exceeds X GB per core/slot --
#BSUB -M 1.1GB
### -- set walltime limit: hh:mm --
#BSUB -W 4:00
### -- set the email address --
#BSUB -u mberos@dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion --
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./Logs/Unzip_%J.out
#BSUB -e ./Logs/Unzip_%J.err
# here follow the commands you want to execute with input.in as the input file

# Actually, it could be more simple to unzip using * for finding any characters (regex pattern search)
# By not using -j, you would preserve the folders structure inside a zip file.

# For wildcard matching recursively on several folders, you can do:
# unzip zip_filename.zip '*noh_wy*_cl1344_iter_y-2050/*'

unzip_to="../../Unzipped"

for name in YYYYMMDD-HHMM_Scenario_Results; do

  # Split name string on '_'
  parts=(${name//_/ })

  # Assign middle part to short_name
  short_name=${parts[1]}

  # .csv's
  unzip -j Workflow/OverallResults/${name}.zip Workflow/OverallResults/${short_name}_LOLD.csv -d "${unzip_to}/OverallResults"
  unzip -j Workflow/OverallResults/${name}.zip Workflow/OverallResults/${short_name}_ProcessTime.csv -d "${unzip_to}/OverallResults"
  unzip -j Workflow/OverallResults/${name}.zip Workflow/OverallResults/${short_name}_results.pkl -d "${unzip_to}/OverallResults"
  unzip -j Workflow/OverallResults/${name}.zip Workflow/MetaResults/${short_name}_meta.ini -d "${unzip_to}/OverallResults"

  # Figures
  # unzip -j Workflow/OverallResults/${name}.zip Workflow/OverallResults/${short_name}_AntaresEmissions.html -d "${unzip_to}/OverallResults"
  # unzip -j Workflow/OverallResults/${name}.zip Workflow/OverallResults/${short_name}_BalmorelEmissions.html -d "${unzip_to}/OverallResults"
  # unzip -j Workflow/OverallResults/${name}.zip Workflow/OverallResults/${short_name}_AntaresH2GenerationFuel.html -d "${unzip_to}/OverallResults"
  # unzip -j Workflow/OverallResults/${name}.zip Workflow/OverallResults/${short_name}_BalmorelH2GenerationFuel.html -d "${unzip_to}/OverallResults"
  # unzip -j Workflow/OverallResults/${name}.zip Workflow/OverallResults/${short_name}_AntaresGenerationFuel.html -d "${unzip_to}/OverallResults"
  # unzip -j Workflow/OverallResults/${name}.zip Workflow/OverallResults/${short_name}_BalmorelGenerationFuel.html -d "${unzip_to}/OverallResults"

  # name=20240521-0754_LTCapCredRisk_Results
  # # Split name string on '_'
  # parts=(${name//_/ })

  # Assign middle part to short_name
  short_name=${parts[1]}
  # for iter in 0 1 2 3 4 5 6 7 8 9; do
  #     unzip -j Workflow/OverallResults/${name}.zip Balmorel/LTFictDemFunc1Max/model/MainResults_${short_name}_Iter${iter}.gdx -d "${unzip_to}"
  # done
  unzip -j Workflow/OverallResults/${name}.zip "Balmorel/*" -d "${unzip_to}"

  # Set previous results variable first
  prev=""
  # List ALL Antares files and loop through them (but print only highest level directory, i.e. the top level of Antares/output)
  unzip -l Workflow/OverallResults/${name}.zip | grep -E 'Antares/output/' | awk -F'/' '{print $3}' | while read -r ant_out; do
    # Check if we have already extracted from this folder
    if [ "${prev}" != "${ant_out}" ]; then
      unzip -j Workflow/OverallResults/${name}.zip Antares/output/${ant_out}/annualSystemCost.txt -d "${unzip_to}/Antares/${ant_out}"
    fi
    prev=$ant_out
  done
done

