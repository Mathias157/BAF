"""
Initialisation Script

Created on 18/08/2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

print('\n|--------------------------------------------------|')   
print('              INITIALISATION')
print('|--------------------------------------------------|\n')  

import pandas as pd
import platform
OS = platform.platform().split('-')[0]
import os
import numpy as np
import pickle
import configparser
import sys
from pybalmorel import IncFile, Balmorel

Config = configparser.ConfigParser()
Config.read('Config.ini')
gams_system_directory = Config.get('RunMetaData', 'gams_system_directory')
        
if not('SC' in locals()):
    try:
        # Try to read something from the command line
        SC = sys.argv[1]
    except:
        # Otherwise, read config from top level
        print('Reading SC from Config.ini..')
        SC = Config.get('RunMetaData', 'SC')

# Make scenario specific config file
Config.set('RunMetaData', 'CurrentIter', '0')
with open('Workflow/MetaResults/%s_meta.ini'%SC, 'w') as f:
  Config.write(f)
  
### 0.0 Load configuration file
SC_folder = Config.get('RunMetaData', 'SC_Folder')
ResetReserveMargin = Config.get('PostProcessing', 'ResetReserveMargin').lower() == 'true'

USE_FICTDEM   = Config.getboolean('PostProcessing', 'Fictivedem')
USE_CAPCRED   = Config.getboolean('PostProcessing', 'Capacitycredit')
USE_H2CAPCRED   = Config.getboolean('PostProcessing', 'H2Capacitycredit')
USE_MARKETVAL  = Config.getboolean('PostProcessing', 'Marketvalue')


# Years
Y = np.array(Config.get('RunMetaData', 'Y').split(',')).astype(int)
Y.sort()
Y = Y.astype(str)

### ------------------------------- ###
###        1. Initialisation        ###
### ------------------------------- ###



### 1.0 Checking Whether Necessary Folders Exist
print('\nChecking folders..')
try:
    os.mkdir('Logs')
    print('Created Logs folder')
except FileExistsError:
    print('Logs folder exist')
    
os.chdir('Workflow')
try:
    os.mkdir('MetaResults')
    print('Created MetaResults folder')
except FileExistsError:
    print('MetaResults folder exist')
try:
    os.mkdir('OverallResults')
    print('Created OverallResults folder')
except FileExistsError:
    print('OverallResults folder exist')
os.chdir('../')


### 1.1 Initialise Placeholder Dataframes
# Electricity not served
fENS = pd.DataFrame({}, columns=['Iter', 'Year', 'Region', 'Value (MWh)'])
fENS.to_csv('./Workflow/OverallResults/%s_ElecNotServedMWh.csv'%SC)

# Hydrogen not served
fENSH2 = pd.DataFrame({}, columns=['Iter', 'Year', 'Region', 'Value (MWh)'])
fENSH2.to_csv('./Workflow/OverallResults/%s_H2NotServedMWh.csv'%SC)

# Loss of load durations
fLOLD = pd.DataFrame({}, columns=['Iter', 'Year', 'Region', 'Carrier', 'Value (h)'])
fLOLD.to_csv('./Workflow/OverallResults/%s_LOLD.csv'%SC)

# Antares Technoeconomic Data
fAntTechno = pd.DataFrame({}, columns=['Iter', 'Year', 'Region', 'Tech']).pivot_table(index=['Iter', 'Year', 'Region', 'Tech'])
with open('./Workflow/OverallResults/%s_AT.pkl'%SC, 'wb') as f:
    pickle.dump(fAntTechno, f)

if USE_MARKETVAL:
    # Market value
    fMV = pd.DataFrame({}, columns=['Iter', 'Year', 'Region', 'Tech', 'Value (eur/MWh)'])
    fMV.to_csv('./Workflow/OverallResults/%s_MV.csv'%SC, index=False)

# Reserve Margin
if USE_CAPCRED:
    fRM = pd.DataFrame({}, columns=['Y', 'R', 'Value', 'Iter', 'Carrier'])
    fRM.to_csv('./Workflow/OverallResults/%s_ResMar.csv'%SC, index=False)
    

### 1.2 Reset Balmorel ANTBALM Addon Data
print('\nResetting ANTBALM addon data...')

path = 'Balmorel/%s/data/'%SC_folder

if USE_MARKETVAL:
    # Market Value
    f = IncFile(name='ANTBALM_MARKETVAL',
                prefix="PARAMETER ANTBALM_MARKETVAL(YYY, RRR, GGG)\n;\n",
                body="ANTBALM_MARKETVAL(YYY, RRR, GGG) = 0;",
                path=path)
    f.save()
    print('Cleared ANTBALM_MARKETVAL.inc')

if USE_FICTDEM:  
    # Fictive Electricity Demand
    f = IncFile(name='ANTBALM_FICTDE',
                path=path)
    f.save()
    print('Cleared ANTBALM_FICTDE.inc')

    # Fictive Electricity Demand 'Users' (actually just to have different profiles pr. year)
    f = IncFile(prefix='SET DEUSER  "Electricity demand user groups. Set must include element RESE for holding demand not included in any other user group"\n/\n',
                name='ANTBALM_FICTDEUSER',
                path=path)
    for year in Y:
        f.body += 'FICTIVE_%s\n'%year     
    f.suffix = '/;'
    f.save()
    print('Cleared ANTBALM_FICTDEUSER.inc')

    # Fictive Electricity Demand Profiles
    f = IncFile(name='ANTBALM_FICTDE_VAR_T',
                path=path)
    f.save()
    print('Cleared ANTBALM_FICTDE_VAR_T.inc')

    # Fictive Hydrogen Demand
    f = IncFile(name='ANTBALM_FICTDH2',
                path=path)
    f.save()
    print('Cleared ANTBALM_FICTDH2.inc')

if USE_CAPCRED:
    # Generator Capacity Credits - set to high in first iteration so it does not have an impact
    capcred_first_iter = 100
    f = IncFile(name='ANTBALM_GCAPCRED',
                prefix="PARAMETER ANTBALM_GCAPCRED(YYY, RRR, GGG)\n;\n",
                body="ANTBALM_GCAPCRED(YYY, RRR, GGG) = %d;"%capcred_first_iter,
                path=path)
    f.save()
    print('Cleared ANTBALM_GCAPCRED.inc')

    # Electricity Transmission Capacity Credit
    f = IncFile(name='ANTBALM_XCAPCRED',
                prefix="PARAMETER ANTBALM_XCAPCRED(YYY, IRRRE, IRRRI)\n;\n",
                body='\n'.join(["ANTBALM_XCAPCRED(YYY, IRRRE, IRRRI) = %d;"%capcred_first_iter,
                            "ANTBALM_XCAPCRED(YYY, 'DE4-E', 'DE4-W') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'DE4-W', 'DE4-E') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'DE4-W', 'DE4-S') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'DE4-S', 'DE4-W') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'DE4-N', 'DE4-E') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'DE4-E', 'DE4-N') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'DE4-S', 'DE4-E') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'DE4-E', 'DE4-S') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'DE4-W', 'DE4-N') = 0;", 
                            "ANTBALM_XCAPCRED(YYY, 'DE4-N', 'DE4-W') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'NO1', 'NO2') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'NO2', 'NO1') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'NO1', 'NO5') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'NO5', 'NO1') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'NO5', 'NO2') = 0;",
                            "ANTBALM_XCAPCRED(YYY, 'NO2', 'NO5') = 0;",]),
                path=path)
    f.save()
    print('Cleared ANTBALM_XCAPCRED.inc')

    if USE_H2CAPCRED:
        # Hydrogen Transmission Capacity Credit
        f = IncFile(name='ANTBALM_XH2CAPCRED',
                    prefix="PARAMETER ANTBALM_XH2CAPCRED(YYY, IRRRE, IRRRI)\n;\n",
                    body='\n'.join(["ANTBALM_XH2CAPCRED(YYY, IRRRE, IRRRI) = %d;"%capcred_first_iter,
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-E', 'DE4-W') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-W', 'DE4-E') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-W', 'DE4-S') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-S', 'DE4-W') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-N', 'DE4-E') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-E', 'DE4-N') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-S', 'DE4-E') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-E', 'DE4-S') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-W', 'DE4-N') = 0;", 
                                "ANTBALM_XH2CAPCRED(YYY, 'DE4-N', 'DE4-W') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'NO1', 'NO2') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'NO2', 'NO1') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'NO1', 'NO5') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'NO5', 'NO1') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'NO5', 'NO2') = 0;",
                                "ANTBALM_XH2CAPCRED(YYY, 'NO2', 'NO5') = 0;",]),
                    path=path)
        f.save()
        print('Cleared ANTBALM_XH2CAPCRED.inc')

    if ResetReserveMargin:

        # Get reser
        elreserve_margin_start = Config.getfloat('PostProcessing', 'elresmar0')
        h2reserve_margin_start = Config.getfloat('PostProcessing', 'h2resmar0')

        # Electricity Reserve Margin
        f = IncFile(name='ANTBALM_RESMAR',
                    prefix="""* The reserve margin is the amount of extra capacity relative to maximum estimated demand
    * 1.28 is assumed in the publication below
    * Alimou, Yacine, Nadia Maïzi, Jean-Yves Bourmaud, and Marion Li. “Assessing the Security of Electricity Supply through Multi-Scale Modeling: The TIMES-ANTARES Linking Approach.” Applied Energy, 2020. https://doi.org/10.1016/j.apenergy.2020.115717.\n
    * ...which corresponds to an assertion that 28% higher capacity compared to maximum demand is necessary for a secure system\n
    PARAMETER ANTBALM_RESMAR(YYY, RRR) 'Assumed electricity reserve margin'
    ;\n""",
        path=path)
        f.body = "ANTBALM_RESMAR(YYY, RRR) = %0.4f;"%elreserve_margin_start
        f.save()
        print('Cleared ANTBALM_RESMAR.inc')
        
        # Hydrogen Reserve Margin
        f = IncFile(name='ANTBALM_H2RESMAR',
                    prefix="""PARAMETER ANTBALM_H2RESMAR(YYY, RRR) 'Assumed hydrogen reserve margin';\n""",
                    path=path)
        f.body = "ANTBALM_H2RESMAR(YYY, RRR) = %0.4f;"%h2reserve_margin_start
        f.save()
        print('Cleared ANTBALM_H2RESMAR.inc')

# # Minimum Electricity Transmission constraint
# f = IncFile(name='ANTBALM_XMIN', 
#             prefix="* Max transmission investments of 10 GW / decade (assuming model years in Y are separated by 10 years)\nVXKN.UP(Y,IRRRE,IRRRI) = 10000;\nVXH2KN.UP(Y,IRRRE,IRRRI) = 10000;",
#             path=path)
# if USE_MARKETVAL:
#     f.suffix = "\n*Upper limit on tech investments for market value feasibility\nVGKN.UP(Y,IA,G) = 100000000;"
# f.save()
# print('Cleared ANTBALM_XMIN.inc')


### 1.3 Setting Temporal Resolution
print('\nSetting yearly resolution to %s'%(str(Y))) 
f = IncFile(name='Y',
            prefix='SET  Y(YYY) "Years in the simulation"\n/\n',
            body=', '.join(list(Y)),
            suffix='\n/;',
            path=path)
f.save()

### 1.4 Load input data
print('\nLoading input data of Balmorel/%s/model...\n'%SC_folder)
m = Balmorel('Balmorel', gams_system_directory=gams_system_directory)
m.load_incfiles(SC_folder, overwrite=True)
del m

print('\n|--------------------------------------------------|')   
print('              END OF INITIALISATION')
print('|--------------------------------------------------|\n')  
