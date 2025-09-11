"""
Dependencies
This script requires the following installations:
- GAMS >37
  https://www.gams.com/download/
- Gams workspace
  https://www.gams.com/41/docs/API_PY_TUTORIAL.html
- Antares Simulator 8.6.1
  https://antares-simulator.org/

See documentation for further instructions
- Documentation/Balmorel_Antares_Soft_Coupling_Framework_Documentation.pdf

Created on 29.03.2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
from matplotlib import rc
from time import time
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from runpy import run_module
import os
if ('Workflow' in __file__) | ('Pre-Processing' in __file__):
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
import sys
import platform
import configparser
from Workflow.Functions.GeneralHelperFunctions import ErrorLog, log_process_time, check_antares_compilation
       
### 0.0 Load configuration file
Config = configparser.ConfigParser()
Config.read('Config.ini')
MCResultsDelete = Config.getboolean('RunMetaData', 'MCResultsDelete')
MaxIteration = Config.getint('RunMetaData', 'MaxIteration')
skip_first_balmorel_exec = Config.getboolean('RunMetaData', 'SkipFirstBalmorelExec')

# Scenario
SC = Config.get('RunMetaData', 'SC')
SC_folder = Config.get('RunMetaData', 'SC_Folder')

# Years
Y = np.array(Config.get('RunMetaData', 'Y').split(',')).astype(int)
Y.sort()
ref_year = Config.getint('RunMetaData', 'ref_year')

### 0.1 Post-Processing
# Soft-Linking Method
USE_MARKETVAL = Config.getboolean('PostProcessing', 'Marketvalue') | Config.getboolean('PostProcessing', 'ProfitDifference')
USE_FICTDEM   = Config.getboolean('PostProcessing', 'Fictivedem')
USE_CAPCRED   = Config.getboolean('PostProcessing', 'Capacitycredit')
USE_H2CAPCRED   = Config.getboolean('PostProcessing', 'H2Capacitycredit')
USE_FLEXDEM = Config.getboolean('PeriProcessing', 'UseFlexibleDemand')

BalmOpts = {'MARKETVAL' : USE_MARKETVAL,
            'FICTDEM' : USE_FICTDEM,
            'CAPCRED' : USE_CAPCRED,
            'H2CAPCRED' : USE_H2CAPCRED,
            'FLEXDEM' : USE_FLEXDEM
            }
BalmCmds = ['--%s=%s'%(key, 'yes') for key in BalmOpts.keys() if BalmOpts[key]]

# Figure out operating system
OS = platform.platform().split('-')[0] # Assuming that linux will be == HPC!

### 0.2 Find Directories 
wk_dir = os.path.dirname(os.path.realpath(__file__)) 
# On HPC (assuming linux = HPC)
if OS == 'Linux':
  gams_path = "/appl/gams/47.6.0"
  antares_path = r'/zhome/c0/2/105719/Desktop/Antares-8.7.0/bin'
# On desktop PC
else:
  gams_path = r"C:\GAMS\45"
  antares_path = r'C:\Program Files\RTE\Antares\8.6.1\bin'

### 0.3 logging
N_errors = 0      # Amount of errors
error_log = ''    # Error log
computational_time = ''

### 0.4 Prepare all commands
# This can be done by using .replace instead of %

### 0.5 Checking if running this script by itself
if np.all(pd.Series(sys.argv).str.find('Master.py') == -1):
    test_mode = True # Set to N if you're running iterations
    print('\n----------------------------\n\n        Test mode ON\n\n----------------------------\n')
else:
    test_mode = False
   
#%% ------------------------------- ###
###        1. Initiate Loop         ###
### ------------------------------- ###

# Get iteration start
StartIter = Config.getint('RunMetaData', 'StartIteration')
i = StartIter

# Make metadata
Config.set('RunMetaData', 'CurrentIter', str(i))
with open('Workflow/MetaResults/%s_meta.ini'%SC, 'w') as f:
  Config.write(f)

if test_mode or i == 0:
  t_start = time()
  out = run_module('Workflow.Initialisation', init_globals={'SC' : SC})
  sys.stdout.flush()  
  t_stop = time()
  log_process_time('Workflow/OverallResults/%s_ProcessTime.csv'%SC, 
                   i, 'Initialisation', t_stop - t_start)

#%% ------------------------------- ###
###      2. Run Workflow Loop       ###
### ------------------------------- ###

convergence = False

while (not(convergence)) & (N_errors == 0) & (i <= MaxIteration):
  # Save new iteration
  Config.set('RunMetaData', 'CurrentIter', str(i))
  with open('Workflow/MetaResults/%s_meta.ini'%SC, 'w') as f:
    Config.write(f)

  ### 2.1 Run Balmorel
  if not(skip_first_balmorel_exec & (i == StartIter)):
    os.chdir(wk_dir + '/Balmorel/%s/model'%SC_folder)
    # On HPC
    if OS == 'Linux':
      balm_cmd = ['gams', 'Balmorel.gms', '--scenario_name=%s_Iter%d'%(SC, i), '--threads=$LSB_DJOB_NUMPROC'] + BalmCmds # For HPC
    # On Desktop
    else:
      balm_cmd = gams_path + '/gams "Balmorel.gms" --scenario_name=%s_Iter%d '%(SC, i) + ' '.join(BalmCmds)
      
    t_start = time()
    if test_mode:
      succes = subprocess.run(balm_cmd, capture_output=True, text=True) # for testing
      print(succes.stdout)
      print(succes.stderr)
    else:
      succes = subprocess.run(balm_cmd) # options for checking output: capture_output=True, text=True
      pass
    t_stop = time()
    os.chdir(wk_dir)
    log_process_time('Workflow/OverallResults/%s_ProcessTime.csv'%SC, 
                    i, 'Balmorel', t_stop - t_start)
    
    # Check for error, and stop if so
    N_errors, error_log = ErrorLog(succes, N_errors, error_log,
                                'ERROR IN BALMOREL EXECUTION')
    if N_errors > 0:
      break

  # Peri-Processing -> Antares executions to optimize all years
  for year in Y:
    
    # Only run reference year in Antares if first iteration
    if year == ref_year and i != 0:
      continue
    
    ### 2.2 Run Peri-Processing
    
    # Check if another instance is running Peri-Processing or if Antares is still compiling 
    compile_finished, N_errors = check_antares_compilation(5*60, 5, N_errors)
    
    # Stop here, if there's an error
    if N_errors > 0:
      break
    
    t_start = time()
    out = run_module('Workflow.Peri-Processing', init_globals={'year' : str(year),
                                                               'SC_name' : SC})        
    sys.stdout.flush()
    t_stop = time()
    log_process_time('Workflow/OverallResults/%s_ProcessTime.csv'%SC, 
                  i, 'Peri-Processing', t_stop - t_start)

    ### 2.3 Run Antares
    ant_run_name = '%s_Iter%d_Y-%d'%(SC, i, year)
    if OS == 'Linux':
        ant_cmd = ['antares-8.6-solver', 'Antares', '-n', ant_run_name, '--parallel'] # Notice: No name in this case!
    else:
        ant_cmd = '"' + antares_path + '/antares-8.6-solver" -i "Antares" -n %s'%ant_run_name + ' --parallel'
    
    t_start = time()
    if test_mode:
      succes = subprocess.run(ant_cmd, capture_output=True, text=True) # for testing
      print(succes.stdout)
      print(succes.stderr)
    else:
      succes = subprocess.run(ant_cmd)
    t_stop = time()
    log_process_time('Workflow/OverallResults/%s_ProcessTime.csv'%SC, 
                  i, 'Antares-Y%d'%year, t_stop - t_start)
        
    N_errors, error_log = ErrorLog(succes, N_errors, error_log,
                                 'ERROR IN ANTARES EXECUTION, YEAR %d'%year)

    # Stop here, if there's an error
    if N_errors > 0:
      break

  # Stop here, if there's an error
  if N_errors > 0:
    break
  
  
  ### 2.4 Assess convergence
  t_start = time()
  out = run_module('Workflow.ConvergenceCriterion', init_globals={'SC' : SC})
  sys.stdout.flush()
  t_stop = time()
  log_process_time('Workflow/OverallResults/%s_ProcessTime.csv'%SC, 
                  i, 'ConvergenceCriterion', t_stop - t_start)

  convergence = out['convergence']
    
    
  ### 2.5 Start new iteration
  i += 1
  
  
  ### 2.6 Run Post-Processing
  if not(convergence):
    t_start = time()
    out = run_module('Workflow.Post-Processing', init_globals={'SC_name' : SC})
    sys.stdout.flush()
    t_stop = time()
    log_process_time('Workflow/OverallResults/%s_ProcessTime.csv'%SC, 
                  i, 'Post-Processing', t_stop - t_start)
    
  ### 2.7 Delete MC Year Results if chosen to
  if MCResultsDelete:
    for year in Y:
      if year == ref_year and i - 1 != 0:
        continue
      
      # Get run from specific year
      ant_output = pd.Series(os.listdir(wk_dir + '/Antares/output'))
      ant_run_name = '%s_Iter%d_Y-%d'%(SC, i-1, year)
      ant_output = ant_output[ant_output.str.find(ant_run_name.lower()) != -1].sort_values(ascending=False).iloc[0] # Get the latest run

      print('\nDeleting outputs from run\n%s\n'%(ant_output))
      shutil.rmtree(wk_dir + '/Antares/output/' + ant_output + '/economy/mc-ind')
      print('\nDeletion done\n')    
    
print('Subprocess errors: %d'%N_errors)
print(error_log)

#%% ------------------------------- ###
###           3. Analysis           ###
### ------------------------------- ###
 
if N_errors == 0:
  run_module('Workflow.Analysis', init_globals={'SC' : SC})
  sys.stdout.flush()

  os.chdir(wk_dir)
  
