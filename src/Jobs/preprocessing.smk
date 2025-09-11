# Get configurations
from configparser import ConfigParser

config = ConfigParser()
config.read('Config.ini')
geographical_scope = config.get('PreProcessing', 'geographical_scope') 

# Command templates (include space in the end for extra arguments)
preprocessing_cli_cmd = "python Workflow/PreProcessing.py "

# Representative final outputs for preprocessing 
rule all:
    input:
        [
            "Antares/input/renewables/series/es/onshore/series.txt",
            "Balmorel/base/data/WND_VAR_T.inc",
            "Balmorel/base/data/WTRRRVAR_T.inc",
            "Balmorel/base/data/done.txt",
            "Balmorel/base/data/DH_VAR_T.inc"
        ]


# Rule for generating mappings
rule generate_mappings:
    output:
        [
            "Pre-Processing/Output/B2A_regi.pkl",
            "Pre-Processing/Output/A2B_regi.pkl"
        ]
    params:
        geo_scope = geographical_scope
    shell:
        preprocessing_cli_cmd + "generate-mappings"

# Rule for generating VRE profiles for Antares
rule generate_antares_vre:
    input:
        "Pre-Processing/Output/A2B_regi.pkl"
    output:
        "Antares/input/renewables/series/es/onshore/series.txt"
    shell:
        preprocessing_cli_cmd + "generate-antares-vre"

# Rule for generating Balmorel timeseries (VRE and exogenous electricity)
rule generate_balmorel_timeseries:
    input:
        [
            "Pre-Processing/Data/IncFile PreSuffixes/WNDFLH.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WNDFLH_OFF.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/SOLEFLH.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WND_VAR_T.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WND_VAR_T_OFF.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/SOLE_VAR_T.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/DE_VAR_T.inc",
        ]
    output:
        [
            "Balmorel/base/data/WNDFLH.inc",
            "Balmorel/base/data/WNDFLH_OFF.inc",
            "Balmorel/base/data/SOLEFLH.inc",
            "Balmorel/base/data/WND_VAR_T.inc",
            "Balmorel/base/data/WND_VAR_T_OFF.inc",
            "Balmorel/base/data/SOLE_VAR_T.inc",
            "Balmorel/base/data/DE_VAR_T.inc",
        ]
    shell:
        preprocessing_cli_cmd + "generate-balmorel-timeseries"

# Rule for generating Balmorel timeseries for hydropower
rule generate_balmorel_hydro:
    input:
        [
            "Pre-Processing/Data/IncFile PreSuffixes/WTRRRFLH.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WTRRSFLH.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WTRRRVAR_T.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WTRRSVAR_S.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/HYRSMAXVOL_G.inc",
        ]
    params:
        geo_scope=geographical_scope
    output:
        [
            "Balmorel/base/data/WTRRRFLH.inc",
            "Balmorel/base/data/WTRRSFLH.inc",
            "Balmorel/base/data/WTRRRVAR_T.inc",
            "Balmorel/base/data/WTRRSVAR_S.inc",
            "Balmorel/base/data/HYRSMAXVOL_G.inc",
        ]
    shell:
        preprocessing_cli_cmd + "generate-balmorel-hydro"

# Rule for generating Balmorel timeseries for heat
rule generate_balmorel_heat_series:
    input:
        [
            "Pre-Processing/Data/IncFile PreSuffixes/DH_VAR_T.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/INDIVUSERS_DH_VAR_T.inc",
        ]
    output:
        [
            "Balmorel/base/data/DH_VAR_T.inc",
            "Balmorel/base/data/INDIVUSERS_DH_VAR_T.inc",
        ]
    shell:
        preprocessing_cli_cmd + "generate-balmorel-heat-series"


# Rule for generating Balmorel timeseries for heat
rule generate_balmorel_annual_heat_adjustment:
    output:
        [
            "Balmorel/base/data/done.txt"
        ]
    shell:
        preprocessing_cli_cmd + "generate-balmorel-annual-heat-adjustment"