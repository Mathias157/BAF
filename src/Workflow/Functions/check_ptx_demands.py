import pandas as pd
from GeneralHelperFunctions import AntaresOutput
from pybalmorel import MainResults
from itertools import product
from configparser import ConfigParser


def get_ptx_results(
    antares_result: str, balmorel_result: str, gams_system_directory: str
):
    # Load Results
    conf = ConfigParser()
    conf.read("Config.ini")
    antout = AntaresOutput(antares_result)
    mr = MainResults(
        balmorel_result,
        paths="Balmorel/base/model",
        system_directory=gams_system_directory,
    )

    ENDO_EL = pd.DataFrame({"Category", "Region", "Value"})

    areas = conf.get("PreProcessing", "geographical_scope").replace(" ", "").split(",")
    commodities = ["HEAT", "HYDROGEN"]

    # Get Antares PtX results
    data = []
    for area, commodity in product(areas, commodities):
        res = antout.load_link_results(
            [area, "_".join([area, commodity])], temporal="annual"
        )["FLOW LIN."]

        data.append(
            {"Category": commodity, "Region": area, "Value": float(res.sum()) / 1e6}
        )

    ENDO_EL = pd.DataFrame(data)

    # Get PtX Results from Balmorel
    balmorel_ptx = (
        mr.get_result("EL_DEMAND_YCR")
        .query('Category in ["ENDOGENOUS_ELECT2HEAT", "ENDO_H2"]')
        .replace({"Category": {"ENDOGENOUS_ELECT2HEAT": "HEAT", "ENDO_H2": "HYDROGEN"}})
        .pivot_table(index=["Category", "Region"], values="Value", aggfunc="sum")
        .reset_index()
    )

    # Get Antares PtX result in nice format
    antares_ptx = ENDO_EL.reset_index().drop(columns="index")

    balmorel_ptx["Model"] = "Balmorel"
    antares_ptx["Model"] = "Antares"

    return balmorel_ptx.loc[
        :, ["Model", "Category", "Region", "Value"]
    ], antares_ptx.loc[:, ["Model", "Category", "Region", "Value"]]


def clustersize_Kmeansinit_tempres_sensitivity(gams_system_directory: str):
    balmorel_results = [
        "MainResults_baf_test_new_fullyear_Iter0.gdx",
        "MainResults_baf_test_new_Iter0.gdx",
    ]
    antares_results = [
        "20250627-2347eco-baf_test_new_clsize1000_nr5_y-2050",
        "20250627-2330eco-baf_test_new_clsize1000_nr4_y-2050",
        "20250627-2314eco-baf_test_new_clsize1000_nr3_y-2050",
        "20250627-2257eco-baf_test_new_clsize1000_nr2_y-2050",
        "20250627-2241eco-baf_test_new_clsize1000_nr1_y-2050",
        "20250627-2224eco-baf_test_new_clsize100_nr5_y-2050",
        "20250627-2219eco-baf_test_new_clsize100_nr4_y-2050",
        "20250627-2215eco-baf_test_new_clsize100_nr3_y-2050",
        "20250627-2210eco-baf_test_new_clsize100_nr2_y-2050",
        "20250627-2205eco-baf_test_new_clsize100_nr1_y-2050",
        "20250627-2200eco-baf_test_new_clsize20_nr5_y-2050",
        "20250627-2158eco-baf_test_new_clsize20_nr4_y-2050",
        "20250627-2155eco-baf_test_new_clsize20_nr3_y-2050",
        "20250627-2152eco-baf_test_new_clsize20_nr2_y-2050",
        "20250627-2149eco-baf_test_new_clsize20_nr1_y-2050",
        "20250627-2146eco-baf_test_new_clsize7_nr5_y-2050",
        "20250627-2144eco-baf_test_new_clsize7_nr4_y-2050",
        "20250627-2141eco-baf_test_new_clsize7_nr3_y-2050",
        "20250627-2139eco-baf_test_new_clsize7_nr2_y-2050",
        "20250627-2136eco-baf_test_new_clsize7_nr1_y-2050",
        "20250627-2134eco-baf_test_new_fullyear_clsize1000_nr5_y-2050",
        "20250627-2045eco-baf_test_new_fullyear_clsize1000_nr4_y-2050",
        "20250627-1954eco-baf_test_new_fullyear_clsize1000_nr3_y-2050",
        "20250627-1903eco-baf_test_new_fullyear_clsize1000_nr2_y-2050",
        "20250627-1815eco-baf_test_new_fullyear_clsize1000_nr1_y-2050",
        "20250627-1722eco-baf_test_new_fullyear_clsize100_nr5_y-2050",
        "20250627-1713eco-baf_test_new_fullyear_clsize100_nr4_y-2050",
        "20250627-1704eco-baf_test_new_fullyear_clsize100_nr3_y-2050",
        "20250627-1655eco-baf_test_new_fullyear_clsize100_nr2_y-2050",
        "20250627-1646eco-baf_test_new_fullyear_clsize100_nr1_y-2050",
        "20250627-1637eco-baf_test_new_fullyear_clsize20_nr5_y-2050",
        "20250627-1633eco-baf_test_new_fullyear_clsize20_nr4_y-2050",
        "20250627-1629eco-baf_test_new_fullyear_clsize20_nr3_y-2050",
        "20250627-1625eco-baf_test_new_fullyear_clsize20_nr2_y-2050",
        "20250627-1620eco-baf_test_new_fullyear_clsize20_nr1_y-2050",
        "20250627-1616eco-baf_test_new_fullyear_clsize7_nr5_y-2050",
        "20250627-1612eco-baf_test_new_fullyear_clsize7_nr4_y-2050",
        "20250627-1609eco-baf_test_new_fullyear_clsize7_nr3_y-2050",
        "20250627-1606eco-baf_test_new_fullyear_clsize7_nr2_y-2050",
        "20250627-1602eco-baf_test_new_fullyear_clsize7_nr1_y-2050",
        "20250718-1530eco-baf_test_new_vectorised_ksmooth",
    ]

    antares_temp = pd.DataFrame({})
    balmorel_temp = pd.DataFrame({})

    for antares_result in antares_results:
        if "fullyear" in antares_result:
            balmorel_result = balmorel_results[0]
            scenario = "fullyear"
        else:
            balmorel_result = balmorel_results[1]
            scenario = "timeslices"
        scenario = "vectorised_ksmooth"

        balmorel_ptx, antares_ptx = get_ptx_results(
            antares_result, balmorel_result, gams_system_directory
        )

        # Get metadata
        # cluster_size = int(antares_result[antares_result.find('clsize'):].split('_')[0].replace('clsize', ''))
        # iteration = int(antares_result[antares_result.find('nr'):].split('_')[0].replace('nr', ''))
        cluster_size = 1248
        iteration = 1

        # Assign metadata
        balmorel_ptx["clustersize"] = cluster_size
        antares_ptx["clustersize"] = cluster_size
        balmorel_ptx["iteration"] = iteration
        antares_ptx["iteration"] = iteration
        balmorel_ptx["scenario"] = scenario
        antares_ptx["scenario"] = scenario

        print("Cluster size: ", cluster_size, "Nr: ", iteration)

        antares_temp = pd.concat((antares_temp, antares_ptx), ignore_index=True)
        balmorel_temp = pd.concat((balmorel_temp, balmorel_ptx), ignore_index=True)

    pd.concat((antares_temp, balmorel_temp), ignore_index=True).to_csv(
        "PtX_demand_comparison.csv"
    )


if __name__ == "__main__":
    gams_system_directory = "/opt/gams/48.5"
    gams_system_directory = "/appl/gams/47.6.0"
    # clustersize_Kmeansinit_tempres_sensitivity(gams_system_directory)

    concated = pd.DataFrame()
    for balmorel_result, antares_result in [
        ["MainResults_EUtest_S4T56_Iter0.gdx", "20250902-2142eco-eutest_s4t56_2_5pctbw_iter0_y-2050"],
        ["MainResults_EUtest_S4T56_Iter0.gdx", "20250901-1609eco-eutest_s4t56_iter0_y-2050"],
        ["MainResults_EUtest_S4T168_Iter0.gdx","20250901-1639eco-eutest_s4t168_iter0_y-2050"],
        ["MainResults_EUFictDem_S4T56_Iter0.gdx", "20250902-1913eco-eufictdem_s4t56_2_5pctbw_iter0_y-2050"],
        ["MainResults_EUFictDem_S4T56_Iter0.gdx","20250830-1817eco-eufictdem_s4t56_iter0_y-2050" ],
    ]:
        balmorel_ptx, antares_ptx = get_ptx_results(
            antares_result,
            balmorel_result,
            gams_system_directory,
        )

        temp_concatenated = pd.concat((balmorel_ptx, antares_ptx), ignore_index=True)
        temp_concatenated['AntaresFile'] = antares_result
        temp_concatenated['BalmorelFile'] = balmorel_result
        concated = pd.concat((concated, temp_concatenated), ignore_index=True)

        print(antares_result)
        print(balmorel_result)
        print("Balmorel:\t%0.0f TWh" % balmorel_ptx.Value.sum())
        print("Antares: \t%0.0f TWh" % antares_ptx.Value.sum())

    concated.to_csv('Workflow/OverallResults/PtX_demand_comparison.csv')
