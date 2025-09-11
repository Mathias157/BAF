"""
Post-Processing

Created on 29/03/2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
# %% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import click
import sys
import configparser
from Functions.Methods import (
    calculate_link_capacity_credits,
    calculate_generator_capacity_credits,
    calculate_h2generator_capacity_credits,
    recalculate_resmar,
    calculate_elmarket_values,
    calculate_h2market_values,
    calculate_antares_elmarket_profits,
    calculate_antares_h2market_profits,
    create_elmarketvaluestrings,
    create_h2marketvaluestrings,
    calculate_elfictdem,
    calculate_h2fictdem,
)
from Functions.GeneralHelperFunctions import IncFile, AntaresOutput
from pybalmorel.utils import symbol_to_df
import pickle
import os
import pandas as pd
import numpy as np
import gams
import platform

OS = platform.platform().split("-")[0]


def context(SC_name: str | None):
    print("\n|--------------------------------------------------|")
    print("              POST-PROCESSING")
    print("|--------------------------------------------------|\n")

    if SC_name == None:
        # Otherwise, read config from top level
        print("Reading SC from Config.ini..")
        Config = configparser.ConfigParser()
        Config.read("Config.ini")
        SC_name = Config.get("RunMetaData", "SC")

    # 0.0 Load configurations
    Config = configparser.ConfigParser()
    Config.read("Workflow/MetaResults/%s_meta.ini" % SC_name)
    SC_folder = Config.get("RunMetaData", "SC_Folder")

    # Years
    Y = np.array(Config.get("RunMetaData", "Y").split(",")).astype(int)
    Y.sort()
    Y = Y.astype(str)
    ref_year = Config.getint("RunMetaData", "ref_year")

    # Get current iteration
    i = Config.getint("RunMetaData", "CurrentIter")

    SC = SC_name + "_Iter%d" % i

    return Config, SC, SC_name, SC_folder, Y, ref_year, i


def old_processing(
    Config: configparser.ConfigParser,
    SC: str,
    SC_name: str,
    SC_folder: str,
    Y: np.ndarray,
    ref_year: int,
    i: int,
):
    wk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    USE_MARKETVAL = Config.getboolean("PostProcessing", "Marketvalue")
    USE_H2MARKETVAL = Config.getboolean("PostProcessing", "H2Marketvalue")
    USE_PROFITDIF = Config.getboolean("PostProcessing", "ProfitDifference")
    USE_FICTDEM = Config.getboolean("PostProcessing", "Fictivedem")
    FICTDEMALLOC = Config.get("PostProcessing", "FictAllocation").lower()
    MAXOREXPECTED = Config.get("PostProcessing", "max_or_expected")
    USE_CAPCRED = Config.getboolean("PostProcessing", "Capacitycredit")
    USE_H2CAPCRED = Config.getboolean("PostProcessing", "H2Capacitycredit")
    update_thermal = Config.getboolean("PostProcessing", "UpdateThermal")
    heggarty_func = Config.get(
        "PostProcessing", "HeggartyFunc"
    ).lower()  # Conservative or Risky (Balanced not made yet)
    UseFlexibleDemand = Config.getboolean(
        "PeriProcessing", "UseFlexibleDemand")
    negative_feedback = Config.getboolean(
        "PostProcessing", "negative_feedback")

    # 0.1 Checking if running this script by itself
    if __name__ == "__main__":
        test_mode = "Y"  # Set to N if you're running iterations
        print(
            "\n----------------------------\n\n        Test mode ON\n\n----------------------------\n"
        )
        wk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    else:
        test_mode = "N"
        wk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Weights on fictive electricity demand from A2B
    with open(wk_dir + "/Pre-Processing/Output/A2B_DE_weights.pkl", "rb") as f:
        A2B_DE_weights = pickle.load(f)
    # Weights on fictive hydrogen demand from A2B
    with open(wk_dir + "/Pre-Processing/Output/A2B_DH2_weights.pkl", "rb") as f:
        A2B_DH2_weights = pickle.load(f)
    # Hydrogen node mappings
    with open(wk_dir + "/Pre-Processing/Output/A2B_regi_h2.pkl", "rb") as f:
        A2B_regi_h2 = pickle.load(f)
    # Electricity node mappings
    with open(wk_dir + "/Pre-Processing/Output/A2B_regi.pkl", "rb") as f:
        A2B_regi = pickle.load(f)
    with open(wk_dir + "/Pre-Processing/Output/B2A_regi.pkl", "rb") as f:
        B2A_regi = pickle.load(f)
    with open(wk_dir + "/Pre-Processing/Output/B2A_regi_h2.pkl", "rb") as f:
        B2A_regi_h2 = pickle.load(f)

    # Get last all_endofmodel
    ws = gams.GamsWorkspace()
    ALLENDOFMODEL = ws.add_database_from_gdx(
        wk_dir + "/Balmorel/%s/model/all_endofmodel.gdx" % SC_folder
    )

    # Iteration Meta and Overall Results
    fENS = pd.read_csv(
        "Workflow/OverallResults/%s_ElecNotServedMWh.csv" % SC_name, index_col=0
    )
    fENSH2 = pd.read_csv(
        "Workflow/OverallResults/%s_H2NotServedMWh.csv" % SC_name, index_col=0
    )
    fLOLD = pd.read_csv("Workflow/OverallResults/%s_LOLD.csv" %
                        SC_name, index_col=0)
    fMV = pd.read_csv("Workflow/OverallResults/%s_MV.csv" % SC_name)
    with open(wk_dir + "/Workflow/OverallResults/%s_AT.pkl" % SC_name, "rb") as f:
        fAntTechno = pickle.load(f)

    if USE_CAPCRED:
        if i != 0:
            with open(
                wk_dir + "/Workflow/OverallResults/%s_CC.pkl" % (SC_name), "rb"
            ) as f:
                # Load capacity credits of the latest iteration
                CC = pickle.load(f)
        else:
            CC = pd.DataFrame(
                index=pd.MultiIndex.from_product([[i], Y, B2A_regi.keys()])
            )

        if USE_H2CAPCRED:
            if i != 0:
                with open(
                    wk_dir +
                        "/Workflow/OverallResults/%s_CCH2.pkl" % (
                            SC_name), "rb"
                ) as f:
                    CCH2 = pickle.load(
                        f
                    )  # Load capacity credits of the latest iteration
            else:
                CCH2 = pd.DataFrame(
                    index=pd.MultiIndex.from_product([[i], Y, B2A_regi.keys()])
                )

        # Load interpolation data (read off Heggarty's PhD)
        interpolation_data = pd.read_csv(
            os.path.abspath(
                "Pre-Processing/Data/HeggartyFunction/%s.csv"
                % heggarty_func.capitalize()
            ),
            header=None,
        )

        # Create databases for reserve margins using the LOLD database structure
        RESMAREL = (
            fLOLD[(fLOLD["Iter"] == i) & (fLOLD.Carrier == "Electricity")]
            .pivot_table(index="Year", columns="Region", values="Value (h)")
            .copy()
        )
        # Change column names to Balmorel names
        RESMAREL.columns = [A2B_regi[col][0] for col in RESMAREL.columns]
        if USE_H2CAPCRED:
            RESMARH2 = (
                fLOLD[(fLOLD["Iter"] == i) & (fLOLD.Carrier == "Hydrogen")]
                .pivot_table(index="Year", columns="Region", values="Value (h)")
                .copy()
            )
            RESMARH2.columns = [A2B_regi_h2[col][0]
                                for col in RESMARH2.columns]

    # 0.2 Technologies transfered from Balmorel, with marginal costs
    with open(wk_dir + "/Pre-Processing/Output/BalmTechs.pkl", "rb") as f:
        BalmTechs = pickle.load(f)

    # %% ------------------------------- ###
    ###        1. Post-Processing       ###
    ### ------------------------------- ###

    # 1.0 Create Balmorel timeseries
    S = (
        ["S0%d" % i for i in range(1, 10)]
        + ["S%d" % i for i in range(10, 53)]
        + ["S53"]
    )
    T = (
        ["T00%d" % i for i in range(1, 10)]
        + ["T0%d" % i for i in range(10, 100)]
        + ["T%d" % i for i in range(100, 169)]
    )
    idx = pd.MultiIndex.from_product([S, T], names=["S", "T"])[:8760]
    hour = pd.DataFrame(data=np.arange(8760), columns=["Hour"], index=idx)

    # 1.1 Load MainResults
    ws = gams.GamsWorkspace()
    db = ws.add_database_from_gdx(
        wk_dir + "/Balmorel/%s/model/MainResults_%s.gdx" % (SC_folder, SC)
    )

    # 1.2 Capacities to dataframe
    cap = symbol_to_df(
        db,
        "G_CAP_YCRAF",
        ["Y", "C", "R", "A", "G", "F", "Commodity", "Tech", "Var", "Unit", "Value"],
    )
    eltran = symbol_to_df(
        db, "X_CAP_YCR", ["Y", "C", "RE", "RI", "Var", "Unit", "Value"]
    )
    h2tran = symbol_to_df(
        db, "XH2_CAP_YCR", ["Y", "C", "RE", "RI", "Var", "Unit", "Value"]
    )

    # Get function for factor on fictive demand
    fict_de_factor = Config.get("PostProcessing", "FictElFactorFunc")
    fict_dh2_factor = Config.get("PostProcessing", "FictH2FactorFunc")

    # Load Balmorel timeseries index (electricity price is most light-weight)
    balm_t = symbol_to_df(
        db, "EL_PRICE_YCRST", ["Y", "C", "R", "S", "T", "Unit", "Val"]
    ).groupby(by=["S", "T"])
    balm_t = balm_t.aggregate({"Val": np.sum}).reset_index()[["S", "T"]]

    # Placeholders for Balmorel Input
    incfiles = {
        "ANTBALM_MAXDEM": IncFile(
            name="ANTBALM_MAXDEM",
            prefix="TABLE ANTBALM_MAXDEM1(RRR, YYY) 'The maximum exogenous electricity demand in all stochastic years of Antares, weighted to corresponding Balmorel regions'\n",
            suffix="\n;\nANTBALM_MAXDEM(YYY,RRR) = ANTBALM_MAXDEM1(RRR, YYY);\nANTBALM_MAXDEM1(RRR, YYY) = 0;",
            body=pd.DataFrame(columns=Y),
            path="Balmorel/%s/data/" % SC_folder,
        ),
        "ANTBALM_H2MAXDEM": IncFile(
            name="ANTBALM_H2MAXDEM",
            prefix="TABLE ANTBALM_H2MAXDEM1(RRR, YYY) 'The maximum exogenous hydrogen demand in all stochastic years of Antares, weighted to corresponding Balmorel regions'\n",
            suffix="\n;\nANTBALM_H2MAXDEM(YYY,RRR) = ANTBALM_H2MAXDEM1(RRR, YYY);\nANTBALM_H2MAXDEM1(RRR, YYY) = 0;",
            body=pd.DataFrame(columns=Y),
            path="Balmorel/%s/data/" % SC_folder,
        ),
    }
    MARKETVAL = "PARAMETER ANTBALM_MARKETVAL(YYY, RRR, GGG)\n;\n"
    CAPCRED_G = "PARAMETER ANTBALM_GCAPCRED(YYY, RRR, GGG)\n;\nANTBALM_GCAPCRED(YYY, RRR, GGG) = 0;\n"
    # Internal DE links do not benefit capacity credit
    CAPCRED_X = "PARAMETER ANTBALM_XCAPCRED(YYY, IRRRE, IRRRI)\n;\nANTBALM_XCAPCRED(YYY, IRRRE, IRRRI) = 0;\n"
    CAPCRED_XH2 = "PARAMETER ANTBALM_XH2CAPCRED(YYY, IRRRE, IRRRI)\n;\nANTBALM_XH2CAPCRED(YYY, IRRRE, IRRRI) = 0;\n"

    FICTDE = ""
    FICTDH2 = ""
    FICTDE_VAR_T = ""
    if USE_FICTDEM:
        if i != 0:
            fDEVAR = pd.read_csv(
                "Workflow/MetaResults/%s_FICTDEprofile.csv" % SC_name, index_col=[0, 1]
            )
            fDH2VAR = pd.read_csv(
                "Workflow/MetaResults/%s_FICTDH2profile.csv" % SC_name, index_col=[0, 1]
            )
        else:
            # Create electricity and hydrogen fictive demand profiles
            fDEVAR = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [Y, B2A_regi.keys()], names=["Year", "Region"]
                ),
                columns=[balm_t.S.unique()],
            )
            fDEVAR.loc[:, :] = 0
            fDH2VAR = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    [Y, B2A_regi.keys()], names=["Year", "Region"]
                ),
                columns=[balm_t.S.unique()],
            )
            fDH2VAR.loc[:, :] = 0
    # 1.3 Get Antares Runs
    l = np.array(os.listdir(wk_dir + "/Antares/output"))
    l.sort()
    l = pd.Series(l[l != "maps"])

    for year in Y:
        # Skip reference year
        if str(ref_year) == year:
            continue

        # Get run from specific year
        idx = l.str.find("eco-" + SC.lower() + "_y-%s" % year) != -1
        ant_res = l[idx].iloc[-1]  # Should be the most recent
        print("Analysing Antares result %s\n" % ant_res)
        # Get run
        AntOut = AntaresOutput(ant_res, wk_dir=wk_dir)

        if UseFlexibleDemand:
            flexdem_uns = AntOut.load_area_results("0_flexdem", "details")

        for area in A2B_regi.keys():
            if not (area in ["ITCO"]):
                # 1.4 Load Antares Results
                f = AntOut.load_area_results(area)

                # Unsupplied Energy
                UNSENR_arr = f.loc[
                    :, "UNSP. ENRG"
                ]  # Hourly expected unsupplied energy for DE_VAR_T (average)
                # UNSENR_arr = AntOut.collect_mcyears('UNSP. ENRG', area).quantile(.5, axis=1)   # Hourly median unsupplied energy for DE_VAR_T (but we don't care when exactly LOLD happens, so this is wrong)
                if (MAXOREXPECTED.lower() == "max") and (USE_FICTDEM):
                    UNSENR_arr = f.loc[:, "UNSP. ENRG.3"].astype(
                        int
                    )  # Hourly unsupplied energy for DE_VAR_T
                UNSENR_arr.index = np.arange(
                    0, len(UNSENR_arr))  # Correct index

                if UseFlexibleDemand:
                    # Add loss of load for flexible demand (if demand is not satisfied within the week)
                    UNSENR_arr += flexdem_uns["%s_flexloss" % area]

                t = hour.reset_index()
                t.loc[: len(UNSENR_arr) - 1, "UNSENR"] = UNSENR_arr.values
                t = t.fillna(0)

                # Store expected unsupplied electrical energy
                fENS = pd.concat(
                    (
                        fENS,
                        pd.DataFrame(
                            {
                                "Iter": i,
                                "Year": year,
                                "Region": area,
                                "Value (MWh)": UNSENR_arr.sum(),
                            },
                            index=[0],
                        ),
                    ),
                    ignore_index=True,
                )

                if USE_CAPCRED | USE_MARKETVAL | USE_PROFITDIF:
                    # Load
                    AntOut.load = AntOut.collect_mcyears(
                        "LOAD", area
                    )  # Loads from all mc-years
                    max_load = AntOut.load.max().max()

                    # ELCC Analysis
                    # LOLD = f.loc[:, 'LOLD']
                    # data = pd.DataFrame({'Load' : load/1e3,
                    #                      'ENS' : UNSENR_arr/1e3}).sort_values(by='Load')
                    # data.index = np.arange(len(data))

                    # max_vals = pd.DataFrame({'Load' : [], 'ENS' : []})
                    # n = 0
                    # for i in range(len(data)):
                    #     if (i == 0):
                    #         max_vals.loc[n, 'Load'] = data.loc[i, 'Load']
                    #         max_vals.loc[n, 'ENS'] = data.loc[i, 'ENS']
                    #         n += 1
                    #     elif (data.loc[i, 'ENS'] > max_vals.loc[n-1, 'ENS']):
                    #         max_vals.loc[n, 'Load'] = data.loc[i, 'Load']
                    #         max_vals.loc[n, 'ENS'] = data.loc[i, 'ENS']
                    #         n += 1
                    #     elif i == len(data) - 1:
                    #         max_vals.loc[n, 'Load'] = data.loc[i, 'Load']
                    #         max_vals.loc[n, 'ENS'] = max_vals.loc[n-1, 'ENS']

                    # fig, ax = plt.subplots()
                    # ax.plot(data['Load'], data['ENS'], 'r+', markersize=1)
                    # ax.plot(max_vals['Load'], max_vals['ENS'], 'b-', linewidth=1)
                    # ax.set_ylabel('ENS (GWh)')
                    # ax.set_xlabel('Load (GW)')
                    # ax.set_title('%s - %s'%(area, year))

                    # Thermal generation
                    AntOut.therm_gen = {}
                    for tech in BalmTechs.keys():
                        for fuel in BalmTechs[tech].keys():
                            try:
                                # Collect all mc-year results
                                AntOut.therm_gen[tech.lower() + "_" + fuel.lower()] = (
                                    AntOut.collect_mcyears(
                                        tech.lower() + "_" + fuel.lower(),
                                        area,
                                        result_type="details",
                                    )
                                )

                            except (FileNotFoundError, KeyError):
                                # If no thermal generation (FileNotFound) or the specific tech doesn't exist (KeyError)
                                AntOut.therm_gen[tech.lower() + "_" + fuel.lower()] = (
                                    pd.DataFrame(np.zeros(1))
                                )

                    # Renewable Generation
                    AntOut.ren_gen = {}
                    for ren in ["WIND ONSHORE", "SOLAR PV"]:
                        AntOut.ren_gen[ren] = AntOut.collect_mcyears(ren, area)

                # Go through corresponding Balmorel areas
                for BalmArea in A2B_regi[area]:
                    # Weight
                    weight = A2B_DE_weights[area][BalmArea]

                    if USE_MARKETVAL:
                        MARKETVAL, fMV = calculate_elmarket_values(
                            i, area, BalmArea, year, MARKETVAL, fMV, AntOut
                        )

                    if USE_PROFITDIF:
                        antares_profits_pr_mwh = calculate_antares_elmarket_profits(
                            i, area, year, AntOut, fAntTechno
                        )

                        # Make market value string
                        MARKETVAL, fMV = create_elmarketvaluestrings(
                            antares_profits_pr_mwh, i, year, BalmArea, MARKETVAL, fMV
                        )

                    if USE_CAPCRED:
                        CC, CAPCRED_G = calculate_generator_capacity_credits(
                            i,
                            BalmArea,
                            area,
                            year,
                            AntOut,
                            cap,
                            A2B_regi,
                            CC,
                            update_thermal,
                            CAPCRED_G,
                        )

                        # Save maximum demand from area
                        incfiles["ANTBALM_MAXDEM"].body.loc[BalmArea, year] = (
                            weight * max_load
                        )

                        # Update reserve margin
                        get_lold = fLOLD.pivot_table(
                            index=["Iter", "Year", "Region"],
                            columns="Carrier",
                            values="Value (h)",
                        )
                        get_lold = get_lold.loc[(
                            i, int(year), area), "Electricity"]
                        RESMAREL = recalculate_resmar(
                            ALLENDOFMODEL,
                            year,
                            BalmArea,
                            "Electricity",
                            heggarty_func,
                            interpolation_data,
                            RESMAREL,
                            get_lold,
                        )

                    if USE_FICTDEM:
                        # Load max if chosen to use max, otherwise keep expected

                        get_lold = fLOLD.pivot_table(
                            index=["Iter", "Year", "Region"],
                            columns="Carrier",
                            values="Value (h)",
                        )
                        get_lold = get_lold.loc[(
                            i, int(year), area), "Electricity"]
                        fDEVAR, FICTDE = calculate_elfictdem(
                            FICTDEMALLOC,
                            balm_t,
                            t,
                            BalmArea,
                            weight,
                            year,
                            fict_de_factor,
                            fDEVAR,
                            UNSENR_arr,
                            FICTDE,
                            get_lold,
                            fENS,
                            i,
                            B2A_regi,
                            negative_feedback,
                        )

                if USE_CAPCRED:
                    CC, CAPCRED_X = calculate_link_capacity_credits(
                        i, area, year, AntOut, eltran, CC, A2B_regi, CAPCRED_X
                    )

        # Hydrogen areas
        # Hence, use demand node
        for area in A2B_regi_h2.keys():
            # Unserved Energy
            f = pd.read_table(
                wk_dir
                + "/Antares/output/"
                + ant_res
                + "/economy/mc-all/areas/%s/values-hourly.txt" % area.lower(),
                skiprows=[0, 1, 2, 3, 5, 6],
            )

            UNSENR_arr = f.loc[:, "UNSP. ENRG"].astype(
                int
            )  # Hourly unsupplied energy for DH2_VAR_T
            UNSENR_arr.index = np.arange(0, len(UNSENR_arr))  # Correct index
            t = hour.reset_index()
            t.loc[: len(UNSENR_arr) - 1, "UNSENR"] = UNSENR_arr.values
            t = t.fillna(0)

            # Store expected unserved energy
            fENSH2 = pd.concat(
                (
                    fENSH2,
                    pd.DataFrame(
                        {
                            "Iter": i,
                            "Year": year,
                            "Region": area,
                            "Value (MWh)": UNSENR_arr.sum(),
                        },
                        index=[0],
                    ),
                ),
                ignore_index=True,
            )

            if USE_FICTDEM:
                # Load max if chosen to use max, otherwise keep expected
                if MAXOREXPECTED.lower() == "max":
                    UNSENR_arr = f.loc[:, "UNSP. ENRG.3"].astype(
                        int
                    )  # Hourly unsupplied energy for DH2_VAR_T
                    UNSENR_arr.index = np.arange(
                        0, len(UNSENR_arr))  # Correct index

                get_lold = fLOLD.pivot_table(
                    index=["Iter", "Year", "Region"],
                    columns="Carrier",
                    values="Value (h)",
                )
                get_lold = get_lold.loc[(i, int(year), area), "Hydrogen"]
                fDH2VAR, FICTDH2 = calculate_h2fictdem(
                    FICTDEMALLOC,
                    balm_t,
                    t,
                    BalmArea,
                    area,
                    year,
                    A2B_regi_h2,
                    A2B_DH2_weights,
                    fict_dh2_factor,
                    fDH2VAR,
                    UNSENR_arr,
                    FICTDH2,
                    get_lold,
                    fENSH2,
                    i,
                    B2A_regi_h2,
                    negative_feedback,
                )

            if (
                (USE_MARKETVAL and USE_H2MARKETVAL)
                or (USE_CAPCRED and USE_H2CAPCRED)
                or USE_PROFITDIF
            ):
                # Load
                AntOut.load = AntOut.collect_mcyears(
                    "LOAD", area
                )  # Loads from all mc-years
                max_load = AntOut.load.max().max()

                # Thermal generation
                AntOut.therm_gen = {}
                for tech in [
                    "steam-methane-reforming-ccs_natgas",
                    "steam-methane-reforming_natgas",
                ]:
                    try:
                        # Collect all mc-year results
                        AntOut.therm_gen[tech] = AntOut.collect_mcyears(
                            tech.lower(), area, result_type="details"
                        )

                    except (FileNotFoundError, KeyError):
                        # If no thermal generation (FileNotFound) or the specific tech doesn't exist (KeyError)
                        AntOut.therm_gen[tech] = pd.DataFrame(np.zeros(1))

                for BalmArea in A2B_regi_h2[area]:
                    # Weight
                    weight = A2B_DH2_weights[area][BalmArea]

                    if USE_H2MARKETVAL and USE_MARKETVAL:
                        MARKETVAL, fMV = calculate_h2market_values(
                            i, area, BalmArea, year, MARKETVAL, fMV, AntOut
                        )

                    if USE_PROFITDIF:
                        antares_profits_pr_mwh = calculate_antares_h2market_profits(
                            i, area, year, AntOut, fAntTechno
                        )

                        # Make market value string
                        MARKETVAL, fMV = create_h2marketvaluestrings(
                            antares_profits_pr_mwh, i, year, BalmArea, MARKETVAL, fMV
                        )

                    if USE_H2CAPCRED and USE_CAPCRED:
                        CCH2, CAPCRED_G = calculate_h2generator_capacity_credits(
                            i,
                            BalmArea,
                            area,
                            year,
                            AntOut,
                            cap,
                            A2B_regi_h2,
                            CCH2,
                            update_thermal,
                            CAPCRED_G,
                        )

                        # Save maximum demand from area
                        incfiles["ANTBALM_H2MAXDEM"].body.loc[BalmArea, year] = (
                            weight * max_load
                        )

                        # Update reserve margin
                        get_lold = fLOLD.pivot_table(
                            index=["Iter", "Year", "Region"],
                            columns="Carrier",
                            values="Value (h)",
                        )
                        get_lold = get_lold.loc[(
                            i, int(year), area), "Hydrogen"]
                        RESMARH2 = recalculate_resmar(
                            ALLENDOFMODEL,
                            year,
                            BalmArea,
                            "Hydrogen",
                            heggarty_func,
                            interpolation_data,
                            RESMARH2,
                            get_lold,
                        )

                if USE_H2CAPCRED and USE_CAPCRED:
                    CCH2, CAPCRED_XH2 = calculate_link_capacity_credits(
                        i,
                        area,
                        year,
                        AntOut,
                        h2tran,
                        CCH2,
                        A2B_regi_h2,
                        CAPCRED_XH2,
                        "mean_all",
                        "XH2",
                    )

        print("\n")

    if USE_MARKETVAL | USE_PROFITDIF:
        with open(
            wk_dir + "/Balmorel/%s/data/ANTBALM_MARKETVAL.inc" % SC_folder, "w"
        ) as f:
            f.write(MARKETVAL)

    if USE_FICTDEM:
        # One profile pr year
        for year in Y:
            for area in A2B_regi.keys():
                if not (area in ["ITCO"]):
                    for BalmArea in A2B_regi[area]:
                        if FICTDEMALLOC == "lole_ts":
                            for season in fDEVAR.columns.get_level_values(0):
                                FICTDE_VAR_T = (
                                    FICTDE_VAR_T
                                    + "DE_VAR_T('%s', 'FICTIVE_%s', '%s', TTT) = %d/168;\n"
                                    % (
                                        BalmArea,
                                        year,
                                        season,
                                        fDEVAR.loc[(year, BalmArea), season],
                                    )
                                )
                        elif FICTDEMALLOC == "existing_ts":
                            FICTDE_VAR_T = (
                                FICTDE_VAR_T
                                + "DE_VAR_T('%s', 'FICTIVE_%s', SSS, TTT) = DE_VAR_T('%s', 'RESE', SSS, TTT);\n"
                                % (BalmArea, year, BalmArea)
                            )

        with open(
            wk_dir + "/Balmorel/%s/data/ANTBALM_FICTDE.inc" % SC_folder, "w"
        ) as f:
            f.write(FICTDE)

        with open(
            wk_dir + "/Balmorel/%s/data/ANTBALM_FICTDH2.inc" % SC_folder, "w"
        ) as f:
            f.write(FICTDH2)

        with open(
            wk_dir + "/Balmorel/%s/data/ANTBALM_FICTDE_VAR_T.inc" % SC_folder, "w"
        ) as f:
            f.write(FICTDE_VAR_T)

    if USE_CAPCRED:
        with open(
            wk_dir + "/Balmorel/%s/data/ANTBALM_GCAPCRED.inc" % SC_folder, "w"
        ) as f:
            f.write(CAPCRED_G)

        # Take care of internal links
        CAPCRED_X = (
            CAPCRED_X
            + "ANTBALM_XCAPCRED(YYY, 'DE4-E', 'DE4-W') = 0;\nANTBALM_XCAPCRED(YYY, 'DE4-W', 'DE4-E') = 0;\nANTBALM_XCAPCRED(YYY, 'DE4-W', 'DE4-S') = 0;\nANTBALM_XCAPCRED(YYY, 'DE4-S', 'DE4-W') = 0;\nANTBALM_XCAPCRED(YYY, 'DE4-N', 'DE4-E') = 0;\nANTBALM_XCAPCRED(YYY, 'DE4-E', 'DE4-N') = 0;\nANTBALM_XCAPCRED(YYY, 'DE4-S', 'DE4-E') = 0;\nANTBALM_XCAPCRED(YYY, 'DE4-E', 'DE4-S') = 0;\nANTBALM_XCAPCRED(YYY, 'DE4-W', 'DE4-N') = 0;\nANTBALM_XCAPCRED(YYY, 'DE4-N', 'DE4-W') = 0;\n"
        )
        with open(
            wk_dir + "/Balmorel/%s/data/ANTBALM_XCAPCRED.inc" % SC_folder, "w"
        ) as f:
            f.write(CAPCRED_X)

        with open(wk_dir + "/Workflow/OverallResults/%s_CC.pkl" % SC_name, "wb") as f:
            # Using pickle, as csv gets the types wrong in the index
            pickle.dump(CC, f)

        if USE_H2CAPCRED:
            # Take care of internal links
            CAPCRED_XH2 = (
                CAPCRED_XH2
                + "ANTBALM_XH2CAPCRED(YYY, 'DE4-E', 'DE4-W') = 0;\nANTBALM_XH2CAPCRED(YYY, 'DE4-W', 'DE4-E') = 0;\nANTBALM_XH2CAPCRED(YYY, 'DE4-W', 'DE4-S') = 0;\nANTBALM_XH2CAPCRED(YYY, 'DE4-S', 'DE4-W') = 0;\nANTBALM_XH2CAPCRED(YYY, 'DE4-N', 'DE4-E') = 0;\nANTBALM_XH2CAPCRED(YYY, 'DE4-E', 'DE4-N') = 0;\nANTBALM_XH2CAPCRED(YYY, 'DE4-S', 'DE4-E') = 0;\nANTBALM_XH2CAPCRED(YYY, 'DE4-E', 'DE4-S') = 0;\nANTBALM_XH2CAPCRED(YYY, 'DE4-W', 'DE4-N') = 0;\nANTBALM_XH2CAPCRED(YYY, 'DE4-N', 'DE4-W') = 0;\n"
            )

            with open(
                wk_dir + "/Balmorel/%s/data/ANTBALM_XH2CAPCRED.inc" % SC_folder, "w"
            ) as f:
                f.write(CAPCRED_XH2)

            with open(
                wk_dir + "/Workflow/OverallResults/%s_CCH2.pkl" % SC_name, "wb"
            ) as f:
                pickle.dump(
                    CCH2, f
                )  # Using pickle, as csv gets the types wrong in the index

        # Write reserve margins
        RESMAREL.columns.name = ""
        RESMAREL.index.name = ""
        with open(
            wk_dir + "/Balmorel/%s/data/ANTBALM_RESMAR.inc" % SC_folder, "w"
        ) as f:
            f.write(
                "TABLE ANTBALM_RESMAR(YYY, RRR) 'Assumed electricity reserve margin'\n"
            )
            f.write(RESMAREL.to_string())
            f.write("\n;")

        # Max demand
        incfiles["ANTBALM_MAXDEM"].body.loc[:, str(ref_year)] = (
            0  # Set reference year to zero
        )
        incfiles["ANTBALM_MAXDEM"].body = incfiles["ANTBALM_MAXDEM"].body.to_string()
        incfiles["ANTBALM_MAXDEM"].save()

        if USE_H2CAPCRED:
            RESMARH2.columns.name = ""
            RESMARH2.index.name = ""
            with open(
                wk_dir + "/Balmorel/%s/data/ANTBALM_H2RESMAR.inc" % SC_folder, "w"
            ) as f:
                f.write(
                    "TABLE ANTBALM_H2RESMAR(YYY, RRR) 'Assumed hydrogen reserve margin'\n"
                )
                f.write(RESMARH2.to_string())
                f.write("\n;")

            incfiles["ANTBALM_H2MAXDEM"].body.loc[:, str(ref_year)] = (
                0  # Set reference year to zero
            )
            incfiles["ANTBALM_H2MAXDEM"].body = incfiles[
                "ANTBALM_H2MAXDEM"
            ].body.to_string()
            incfiles["ANTBALM_H2MAXDEM"].save()

    # 1.10 Save Results for Next Iteration
    fENS.to_csv("Workflow/OverallResults/%s_ElecNotServedMWh.csv" % SC_name)
    fENSH2.to_csv("Workflow/OverallResults/%s_H2NotServedMWh.csv" % SC_name)
    fMV.to_csv("Workflow/OverallResults/%s_MV.csv" % SC_name, index=False)
    if USE_FICTDEM:
        fDEVAR.to_csv("Workflow/MetaResults/%s_FICTDEprofile.csv" % SC_name)
        fDH2VAR.to_csv("Workflow/MetaResults/%s_FICTDH2profile.csv" % SC_name)

    print("\n|--------------------------------------------------|")
    print("              END OF POST-PROCESSING")
    print("|--------------------------------------------------|\n")


@click.command()
@click.argument("scenario_name", type=str, required=False, default=None)
def post_process(scenario_name: str | None):
    Config, SC, scenario_name, SC_folder, years, ref_year, iteration = context(
        scenario_name
    )
    old_processing(Config, SC, SC_name, SC_folder, years, ref_year, iteration)
    print(SC, scenario_name, SC_folder, years, ref_year, iteration)


if __name__ == "__main__":
    post_process()
