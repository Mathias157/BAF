# %% ------------------------------- ###
###       0. Script Settings        ###
### ------------------------------- ###
"""
Created on 30/03/2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)

IN ONE SENTENCE:
Transfers data from Balmorel to Antares, runs Antares and saves results

ASSUMPTIONS IN SECTIONS:
- 0.4 Dictionaries are hard-coded, based on current Antares/Balmorel set definitions
      Country list assumes country key is the same for Balmorel+Antares, and that it's in the first 2 letters of the regions
- 1.2 Peak production in VRE series = Peak capacity
- 1.4 Full transmission capacity available all hours
- 1.5 Taking all electricity demand as INFLEXIBLE electricity demand in Antares
- 4.5 Transmission capacity is rather arbitrarily increased by 1 + difference from congestion rate and 0.5
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gams
import subprocess
import os
import sys
import platform
import pickle
import shutil

### 0.0 Plot settings
style = "report"

if style == "report":
    plt.style.use("default")
    fc = "white"
elif style == "ppt":
    plt.style.use("dark_background")
    fc = "none"

### 0.1 Figure out operating system
OS = platform.platform().split("-")[0]  # Assuming that linux will be == HPC!


### 0.2 Checking if running this script by itself
if np.all(pd.Series(sys.argv).str.find("balmorel-balmorel.py") == -1):
    test_mode = "Y"  # Set to N if you're running iterations
    print(
        "\n----------------------------\n\nTest mode ON\n\n----------------------------\n"
    )
else:
    test_mode = "N"
    # print('\n\n\nTest mode off\n\n\n')


### 0.3 Find working directory
wk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if test_mode == "N":
    gams_path = sys.argv[1]
else:
    gams_path = r"C:\GAMS\41"
ant_study = "/BZModelSMALLDKDE"  # the specific antares study

### 0.4 Technologies transfered from Balmorel, with marginal costs
kgGJ2tonMWh = 3.6 / 1e3  # Conversion from kg/GJ to ton/MWh
BalmTechs = {
    "CHP-BACK-PRESSURE": {
        "NATGAS": {"CO2": 56.1 * kgGJ2tonMWh, "MC": 40},
        "WOODCHIPS": {"CO2": 0, "MC": 20},
        "BIOGAS": {"CO2": 0, "MC": 10},
        "MUNIWASTE": {"CO2": 0, "MC": 20},
    },
    "CHP-EXTRACTION": {
        "NATGAS": {"CO2": 56.1 * kgGJ2tonMWh, "MC": 40},
        "WOODCHIPS": {"CO2": 0, "MC": 20},
        "BIOGAS": {"CO2": 0, "MC": 10},
        "MUNIWASTE": {"CO2": 0, "MC": 20},
    },
    "CONDENSING": {
        "NATGAS": {"CO2": 56.1 * kgGJ2tonMWh, "MC": 40},
        "WOODCHIPS": {"CO2": 0, "MC": 20},
        "BIOGAS": {"CO2": 0, "MC": 10},
        "MUNIWASTE": {"CO2": 0, "MC": 20},
        "NUCLEAR": {"CO2": 0, "MC": 5},
    },
    "FUELCELL": {"HYDROGEN": {"CO2": 0, "MC": 0}},
}

### 0.4 Hard-coded dictionaries for Balmorel/Antares set translation

# Fuels
# B2A_fuel = {'WOODCHIPS' : 'woodchips',
#             'BIOGAS' : 'biogas',
#             }
# A2B_fuel = {B2A_fuel[k] : k for k in B2A_fuel.keys()}

# Technologies
# B2A_tech = {'CHP-BACK-PRESSURE' : 'CHP-BP',
#             'CONDENSING' : 'Cond',
#             'CHP-EXTRACTION' }
# A list from DE-DK output: (GW)
# CHP-BACK-PRESSURE LIGHTOIL 0.0
# CHP-BACK-PRESSURE NATGAS 2.220446049250313e-16
# CHP-BACK-PRESSURE STRAW 0.0
# CHP-BACK-PRESSURE WOODCHIPS 2.292637202895013
# CHP-BACK-PRESSURE WOODPELLETS 0.0
# CHP-BACK-PRESSURE BIOGAS 10.86009872138417
# CHP-BACK-PRESSURE COAL 0.0
# CHP-BACK-PRESSURE MUNIWASTE 2.390979571976841
# CHP-BACK-PRESSURE FUELOIL 0.0
# CHP-BACK-PRESSURE LIGNITE 0.0
# CONDENSING LIGHTOIL 0.0
# CONDENSING NATGAS 0.0
# CONDENSING BIOGAS 7.771561172376096e-16
# CONDENSING COAL 0.0
# CONDENSING MUNIWASTE 0.0
# CONDENSING FUELOIL 0.0
# CONDENSING WASTEHEAT 0.0
# CONDENSING LIGNITE 0.0
# CONDENSING NUCLEAR 0.0
# CHP-EXTRACTION NATGAS 0.0
# CHP-EXTRACTION WOODPELLETS 0.0
# CHP-EXTRACTION BIOGAS 0.0
# CHP-EXTRACTION COAL 0.0
# CHP-EXTRACTION LIGNITE 0.0
# SOLAR-PV SUN 353.80588155731743
# HYDRO-RUN-OF-RIVER WATER 8.849395
# WIND-ON WIND 111.19999999999996
# WIND-OFF WIND 113.59742761200643
# INTRASEASONAL-ELECT-STORAGE ELECTRIC 31.555227736978406
# FUELCELL HYDROGEN 4.07380644459292
# HYDRO-RESERVOIRS WATER 0.124


# Renewables
B2A_ren = {"SOLAR-PV": "solar", "WIND-ON": "wind", "WIND-OFF": "wind"}
A2B_tech = {B2A_ren[k]: k for k in B2A_ren.keys()}

# Regions
with open(wk_dir + "/Pre-Processing/B2A_regi.pkl", "rb") as f:
    B2A_regi = pickle.load(f)
with open(wk_dir + "/Pre-Processing/B2A_regi_h2.pkl", "rb") as f:
    B2A_regi_h2 = pickle.load(f)

with open(wk_dir + "/Pre-Processing/A2B_regi.pkl", "rb") as f:
    A2B_regi = pickle.load(f)
with open(wk_dir + "/Pre-Processing/A2B_regi_h2.pkl", "rb") as f:
    A2B_regi_h2 = pickle.load(f)
with open(wk_dir + "/Pre-Processing/A2B_regi_h2_dem.pkl", "rb") as f:
    A2B_regi_h2_dem = pickle.load(f)

# Full antares region list
ANTREGLIST = pd.Series(
    [
        "5_DE00_SRES",
        "5_DKE1_SRES",
        "5_DKW1_SRES",
        #   '6_DE00_SRES',
        #   '6_DKE1_SRES',
        #   '6_DKW1_SRES',
        "7_DE00_SRES",
        "7_DKE1_SRES",
        "7_DKW1_SRES",
        #   '8_DE00_SRES',
        #   '8_DKE1_SRES',
        #   '8_DKW1_SRES',
        "DE00",
        #   'DEKF',
        "DKE1",
        #   'DKKF',
        "DKW1",
    ]
).str.lower()

# Countries
C = pd.Series(list(B2A_regi.keys())).str[:2].unique()

# Weights on fictive electricity demand from A2B
with open(wk_dir + "/Pre-Processing/A2B_DE_weights.pkl", "rb") as f:
    A2B_DE_weights = pickle.load(f)
# Weights on fictive hydrogen demand from A2B
with open(wk_dir + "/Pre-Processing/A2B_DH2_weights.pkl", "rb") as f:
    A2B_DH2_weights = pickle.load(f)
# WEIGHTS FOR THERMAL CAPACITIES B2A
B2A_CH2_weights = {
    "DK1": {"z_h2_c3_dkw1": 1},
    "DK2": {"z_h2_c3_dke1": 1},
    "DE4-N": {"z_h2_c3_de00": 1},
    "DE4-E": {"z_h2_c3_de00": 1},
    "DE4-S": {"z_h2_c3_de00": 1},
    "DE4-W": {"z_h2_c3_de00": 1},
}

# Base factor on fictive electricity demand
fict_de_factor = 1

# GDATA
GDATA = pd.read_excel(wk_dir + "/Pre-Processing/GDATA.xlsx")

### 0.5 Iteration Data
i = int(open("i.txt").readline())

## Scenario
if test_mode == "N":
    SC_name = sys.argv[2]
    SC = SC_name + "_Iter%d" % i
else:
    SC_name = "BalmVal"  # For testing
    SC = SC_name + "_Iter0"

print("\n----------------PERI-PROCESSING---------------\n")
# print(SC_name, '\n', wk_dir, os.path.dirname(os.path.realpath(__file__)) )

## Iteration Meta and Overall Results
# res = pd.read_csv('OverallResults/%s_Result.csv'%SC_name, index_col=0)
# fCAP = pd.read_csv('OverallResults/%s_TotalCapacitiesGW.csv'%SC_name, index_col=0)
# fCAP.loc[i,:] = np.zeros(fCAP.shape[1])
fENS = pd.read_csv("OverallResults/%s_ElecNotServedMWh.csv" % SC_name, index_col=0)
fENS.loc[i, :] = np.zeros(fENS.shape[1])
fENSH2 = pd.read_csv("OverallResults/%s_H2NotServedMWh.csv" % SC_name, index_col=0)
fENSH2.loc[i, :] = np.zeros(fENSH2.shape[1])
fLOLD = pd.read_csv("OverallResults/%s_ElecLOLD.csv" % SC_name, index_col=0)
fLOLD.loc[i, :] = np.zeros(fLOLD.shape[1])
# fDEM = pd.read_csv('OverallResults/%s_DemandTWh.csv'%SC_name, index_col=0)
# fDEM.loc[i,:] = np.zeros(fDEM.shape[1])
if i != 0:
    fDEVAR = pd.read_csv("MetaResults/FICTDEprofile.csv", index_col="S")
    fDH2VAR = pd.read_csv("MetaResults/FICTDH2profile.csv", index_col="S")
else:
    fDEVAR = pd.DataFrame([])
    fDH2VAR = pd.DataFrame([])


### 0.6 Neat Functions
def symbol_to_df(db, symbol, cols="None"):
    """
    Loads a symbol from a GDX database into a pandas dataframe

    Args:
        db (GamsDatabase): The loaded gdx file
        symbol (string): The wanted symbol in the gdx file
        cols (list): The columns
    """
    df = dict((tuple(rec.keys), rec.value) for rec in db[symbol])
    df = pd.DataFrame(df, index=["Value"]).T.reset_index()  # Convert to dataframe
    if cols != "None":
        try:
            df.columns = cols
        except:
            pass
    return df


def B2A_area_equal_weight(area):
    """
    Gets the weight of a Balmorel-to-Antares Parameter

    Returns:
        scalar: weight
    """
    # For a higher resolution in Balmorel,
    # weight is 1 since parameters will be summed
    if len(A2B_regi[area]) > 1:
        weight = 1
    # For a higher resolution in Antares,
    # weight will be 1/(the amount of Antares areas)
    else:
        weight = 1 / len(B2A_regi[A2B_regi[area][0]])
    return weight


def A2B_area_equal_weight(area):
    """
    Gets the weight of a Antares-to-Balmorel Parameter

    Returns:
        scalar: weight
    """
    # For a higher resolution in Antares,
    # weight is 1 since parameters will be summed
    if len(B2A_regi[area]) > 1:
        weight = 1
    # For a higher resolution in Antares,
    # weight will be 1/(the amount of Antares areas)
    else:
        weight = 1 / len(A2B_regi[B2A_regi[area][0]])
    return weight


### 0.7 Example of inputting something from command-line
# print(sys.argv)

# %% ------------------------------- ###
###       1. Peri-Processing        ###
### ------------------------------- ###

### 1.0 Change balopt to operation
balm_path = wk_dir + "/Balmorel/base/model/"
shutil.copyfile(balm_path + "balopt_operation.opt", balm_path + "balopt.opt")


### 1.1 Make full year time-series
shutil.copyfile(
    wk_dir + "/Balmorel/base/data/T_operation.inc", wk_dir + "/Balmorel/base/data/T.inc"
)
shutil.copyfile(
    wk_dir + "/Balmorel/base/data/S_operation.inc", wk_dir + "/Balmorel/base/data/S.inc"
)
shutil.copyfile(
    wk_dir + "/Balmorel/base/data/CHRONOHOUR_operation.inc",
    wk_dir + "/Balmorel/base/data/CHRONOHOUR.inc",
)


print("\nPeri-processing done\n----------------------------------------------\n")

# %% ------------------------------- ###
###    2. Run Balmorel Operation    ###
### ------------------------------- ###

### 2.1 Running Balmorel Operation
os.chdir(balm_path)
if OS == "Linux":
    Balm_cmd = [
        "gams",
        '"%sBalmorel.gms" --scenario_name=%s_Operation' % (balm_path, SC),
    ]  # For HPC, old gams
else:
    Balm_cmd = gams_path + '/gams "%sBalmorel.gms" --scenario_name=%s_Operation' % (
        balm_path,
        SC,
    )
succes = subprocess.run(Balm_cmd)
# print('\n'.join(str(succes.stdout).split('\\r\\n')))


# %% ------------------------------- ###
###        3. Analyse Output        ###
### ------------------------------- ###

print("\n-------------CONVERGENCE-CRITERION------------\n")

### 3.0 Initialisations
l = np.array(os.listdir(balm_path))
l.sort()

### 3.1 Load MainResults
ws = gams.GamsWorkspace()
db = ws.add_database_from_gdx(
    wk_dir + "/Balmorel/base/model/MainResults_%s_Operation.gdx" % SC
)


### 3.2 Production
# pro = symbol_to_df(db, "PRO_YCRAGF", ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity',
#                                        'Tech', 'Unit', 'Value']) # Annual Production, superfluous when loading hourly

pro_t = symbol_to_df(
    db,
    "PRO_YCRAGFST",
    ["Y", "C", "R", "A", "G", "F", "S", "T", "Commodity", "Tech", "Unit", "Value"],
)  # Hourly Production

## Aggregate
idx = pro_t.G.str.find("BACKUP") != -1
ENS = pro_t[idx].groupby(by=["R", "Commodity", "S", "T"]).aggregate({"Value": np.sum})
ENS = ENS.reset_index()

### 3.3 Assess Convergence
for BalmArea in ENS.R.unique():
    idx1 = ENS.R == BalmArea

    for carrier in ENS.loc[idx1, "Commodity"].unique():
        idx2 = ENS.Commodity == carrier

        LOLD = ENS[idx1 & idx2].shape[0]
        print("%s LOLD in %s: \t" % (carrier, BalmArea), LOLD, "h")

        if carrier == "ELECTRICITY":
            fLOLD.loc[i, BalmArea] = LOLD

convergence = np.all(fLOLD.loc[i] <= 3)
print("\nConvergence achieved: %s\n" % convergence)
print(
    "Assessing convergence criterion done\n----------------------------------------------\n"
)

print("\n### ---------------POST-PROCESSING---------------- ###\n")

### 3.4 Create fictive demand
print("\n### Creating fictive demand\n")

# Increase fictive demand if iteration is >20
if i >= 20:
    fict_de_factor = i / 20 * 2

# Load Balmorel timeseries index (electricity price is most light-weight) - this should be the low resolution timeseries though!
print("Loading timeseries from invest run...")
db_inv = ws.add_database_from_gdx(
    wk_dir + "/Balmorel/base/model/MainResults_%s_Invest.gdx" % SC
)

balm_t = symbol_to_df(
    db_inv, "EL_PRICE_YCRST", ["Y", "C", "R", "S", "T", "Unit", "Val"]
).groupby(by=["S", "T"])
balm_t = balm_t.aggregate({"Val": np.sum}).reset_index()[["S", "T"]]
S = ["S0%d" % i for i in range(1, 10)] + ["S%d" % i for i in range(10, 53)] + ["S53"]
T = (
    ["T00%d" % i for i in range(1, 10)]
    + ["T0%d" % i for i in range(10, 100)]
    + ["T%d" % i for i in range(100, 169)]
)
idx = pd.MultiIndex.from_product([S, T], names=["S", "T"])[:8760]
hour = pd.DataFrame(data=np.arange(8760), columns=["Hour"], index=idx)

print("%d seasons:" % len(balm_t.S.unique()), balm_t.S.unique())
print("%d terms:" % len(balm_t["T"].unique()), balm_t["T"].unique())


for BalmArea in ENS.R.unique():
    idx1 = ENS.R == BalmArea

    for carrier in ENS.loc[idx1, "Commodity"].unique():
        idx2 = ENS.Commodity == carrier

        # Get series of LOLE (backup capacity production)
        LOLE = (
            ENS.loc[idx1 & idx2].groupby(by=["S", "T"]).aggregate({"Value": np.sum})
        )  # Not summing, just re-arranging

        if carrier == "ELECTRICITY":
            # Join to full hour
            t = hour.copy()
            t = t.join(LOLE)
            t = t.fillna(0).reset_index()

            front_t = t.copy().drop(columns=["S", "T"])
            front_t.index = np.arange(-len(t), 0)
            back_t = t.copy().drop(columns=["S", "T"])
            back_t.index = np.arange(len(t), 2 * len(t))
            temp_t = pd.concat((front_t, t.copy(), back_t))

            for n, row in balm_t.iterrows():
                # Get current hour
                h1 = temp_t.loc[
                    (temp_t["S"] == row["S"]) & (temp_t["T"] == row["T"]), "Hour"
                ].values[0]

                # Get previous hour
                try:
                    h0 = temp_t.loc[
                        (temp_t["S"] == balm_t.loc[n - 1, "S"])
                        & (temp_t["T"] == balm_t.loc[n - 1, "T"]),
                        "Hour",
                    ].values[0]
                    dif0 = round((h1 - h0) / 2)
                except KeyError:
                    h0 = temp_t.loc[
                        (temp_t["S"] == balm_t.loc[len(balm_t) - 1, "S"])
                        & (temp_t["T"] == balm_t.loc[len(balm_t) - 1, "T"]),
                        "Hour",
                    ].values[0]
                    dif0 = round((h1 + len(t) - h0) / 2)

                # Get last hour
                try:
                    h2 = temp_t.loc[
                        (temp_t["S"] == balm_t.loc[n + 1, "S"])
                        & (temp_t["T"] == balm_t.loc[n + 1, "T"]),
                        "Hour",
                    ].values[0]
                    dif2 = round((h2 - h1) / 2)
                except KeyError:
                    h2 = temp_t.loc[
                        (temp_t["S"] == balm_t.loc[0, "S"])
                        & (temp_t["T"] == balm_t.loc[0, "T"]),
                        "Hour",
                    ].values[0]
                    dif2 = round((h2 + len(t) - h1) / 2)

                # Accumulated Unsupplied Energy
                idx = (temp_t.index >= h1 - dif0) & (temp_t.index < h1 + dif2 - 1)
                balm_t.loc[n, BalmArea + "_UNSELEC"] = temp_t.loc[idx, "Value"].sum()

            agg = balm_t.groupby(by=["S"])
            agg = agg.aggregate({BalmArea + "_UNSELEC": np.sum})

            if i != 0:
                fDEVAR[BalmArea] += agg[BalmArea + "_UNSELEC"] * fict_de_factor
            else:
                fDEVAR[BalmArea] = agg[BalmArea + "_UNSELEC"]

            # Store overall unserved energy
            fENS.loc[i, BalmArea] = agg[BalmArea + "_UNSELEC"].sum()

        else:
            if i != 0:
                fDH2VAR[BalmArea] += (
                    t.groupby(by="S").aggregate({"Value": np.sum}).Value
                )
            else:
                fDH2VAR[BalmArea] = t.groupby(by="S").aggregate({"Value": np.sum}).Value

            # Store overall unserved energy
            fENSH2.loc[i, BalmArea] = (
                LOLE.groupby(by="S").aggregate({"Value": np.sum}).sum().values[0]
            )


### 3.5 Create Balmorel Files
os.chdir(wk_dir + "/Workflow")

FICTDE = ""
FICTDH2 = ""
FICTDE_VAR_T = ""
for BalmArea in ENS.R.unique():
    FICTDE = (
        FICTDE
        + "DE('2050','%s','FICTIVE') = %0.2f;\n" % (BalmArea, fDEVAR[BalmArea].sum())
    )  # <--- Save this in a list or array instead, will accumulate el-demand from electrolyser as well
    FICTDH2 = (
        FICTDH2
        + "HYDROGEN_DH2('2050','%s') = HYDROGEN_DH2('2050','%s') + %0.2f;\n"
        % (BalmArea, BalmArea, fDH2VAR[BalmArea].sum())
    )  # <--- Save this in a list or array instead, will accumulate el-demand from electrolyser as well

    for season in fDEVAR[BalmArea].index:
        FICTDE_VAR_T = (
            FICTDE_VAR_T
            + "DE_VAR_T('%s', 'FICTIVE', '%s', TTT) = %d/168;\n"
            % (BalmArea, season, fDEVAR.loc[season, BalmArea])
        )

# NO MARKET VALUE
# MARKETVAL = MARKETVAL + "\nANTBALM_MARKETVAL(YYY, RRR, GGG) = 0;"
# with open(wk_dir+'/Balmorel/base/data/ANTBALM_MARKETVAL.inc', 'w') as f:
#     f.write(MARKETVAL)

with open(wk_dir + "/Balmorel/base/data/ANTBALM_FICTDE.inc", "w") as f:
    f.write(FICTDE)

with open(wk_dir + "/Balmorel/base/data/ANTBALM_FICTDH2.inc", "w") as f:
    f.write(FICTDH2)

with open(wk_dir + "/Balmorel/base/data/ANTBALM_FICTDE_VAR_T.inc", "w") as f:
    f.write(FICTDE_VAR_T)


### 3.6 Save Results for Next Iteration
# res.to_csv('OverallResults/%s_Result.csv'%SC_name)
# fCAP.to_csv('OverallResults/%s_TotalCapacitiesGW.csv'%SC_name)
fENS.to_csv("OverallResults/%s_ElecNotServedMWh.csv" % SC_name)
fENSH2.to_csv("OverallResults/%s_H2NotServedMWh.csv" % SC_name)
fLOLD.to_csv("OverallResults/%s_ElecLOLD.csv" % SC_name)
# fDEM.to_csv('OverallResults/%s_DemandTWh.csv'%SC_name)
fDEVAR.to_csv("MetaResults/FICTDEprofile.csv")
fDH2VAR.to_csv("MetaResults/FICTDH2profile.csv")
