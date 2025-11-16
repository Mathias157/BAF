import pandas as pd
from GeneralHelperFunctions import AntaresOutput
from pybalmorel import MainResults
from itertools import product
from configparser import ConfigParser
from rich.console import Console 
import click
from pathlib import Path

console = Console()
def get_ptx_results(
    antares_result: str, balmorel_result: str, gams_system_directory: str, scenario_folder: str = 'base', mc_year: str = 'mc-all'
):
    # Load Results
    scenario_name = (
        balmorel_result
        .replace('MainResults_', '')
        .replace('_Iter0.gdx', '')
    )
    conf = ConfigParser()
    conf.read(f"Workflow/MetaResults/{scenario_name}_meta.ini")
    antout = AntaresOutput(antares_result)
    mr = MainResults(
        balmorel_result,
        paths=f"Balmorel/{scenario_folder}/model",
        system_directory=gams_system_directory,
    )

    ENDO_EL = pd.DataFrame({"Category", "Region", "Value"})

    areas = conf.get("PreProcessing", "geographical_scope").replace(" ", "").split(",")
    commodities = ["HEAT", "HYDROGEN"]

    # Get Antares PtX results
    data = []
    for area, commodity in product(areas, commodities):
        res = antout.load_link_results(
            [area, "_".join([area, commodity])], temporal="annual", mc_year=mc_year
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

@click.pass_context
def collect_ptx_results(ctx, antbalm_result_list: list, csv_filename: str):

    gams_system_directory = ctx.obj['gams_system_directory']

    concated = pd.DataFrame()
    for balmorel_result, antares_result in antbalm_result_list:

        # If Balmorel result is list, second argument is scenario folder
        if type(balmorel_result) is list:
            scenario_folder = balmorel_result[1]
            balmorel_result = balmorel_result[0]
        else:
            scenario_folder = 'base'

        # If Antares result is list, second argument is mc year
        if type(antares_result) is list:
            mc_year = antares_result[1]
            antares_result = antares_result[0]
        else:
            mc_year = 'mc-all'

        balmorel_ptx, antares_ptx = get_ptx_results(
                    antares_result,
                    balmorel_result,
                    gams_system_directory,
                    scenario_folder,
                    mc_year
                )

        temp_concatenated = pd.concat((balmorel_ptx, antares_ptx), ignore_index=True)
        temp_concatenated['AntaresFile'] = antares_result
        temp_concatenated['BalmorelFile'] = balmorel_result
        concated = pd.concat((concated, temp_concatenated), ignore_index=True)

        console.log(antares_result)
        console.log(balmorel_result)
        console.log("Balmorel:\t%0.0f TWh" % balmorel_ptx.Value.sum())
        console.log("Antares: \t%0.0f TWh" % antares_ptx.Value.sum())

    concated.to_csv(f'Workflow/OverallResults/PtX_demand_comparison_{csv_filename}.csv')

    return concated


@click.group()
@click.pass_context
@click.option('--gams-directory', type=str, default='/appl/gams/47.6.0/', help='System directory of GAMS')
def CLI(ctx, gams_directory: str):
    
    ctx.ensure_object(dict)
    ctx.obj['gams_system_directory'] = gams_directory

@CLI.command()
@click.pass_context
def clustersize(ctx):
    """
    Sensitivity analyses on the cluster size (amount of supply curves developed)
    compared to the first kernel smoothed tests at small-scale.
    """

    gams_system_directory = ctx.obj['gams_system_directory']

    balmorel_results = [
        "MainResults_baf_test_new_fullyear_Iter0.gdx",
        "MainResults_baf_test_new_Iter0.gdx",
    ]
    antares_results = [
        "20250627-2241eco-baf_test_new_clsize1000_nr1_y-2050",
        "20250627-2205eco-baf_test_new_clsize100_nr1_y-2050",
        "20250627-2149eco-baf_test_new_clsize20_nr1_y-2050",
        "20250627-2136eco-baf_test_new_clsize7_nr1_y-2050",
        "20250627-1815eco-baf_test_new_fullyear_clsize1000_nr1_y-2050",
        "20250627-1646eco-baf_test_new_fullyear_clsize100_nr1_y-2050",
        "20250627-1620eco-baf_test_new_fullyear_clsize20_nr1_y-2050",
        "20250627-1602eco-baf_test_new_fullyear_clsize7_nr1_y-2050",
        "20250707-1359eco-baf_test_new_all_hours_fix_clsize1248_nr1_y-2050",
        "20250718-1530eco-baf_test_new_vectorised_ksmooth",
    ]

    antares_temp = pd.DataFrame({})
    balmorel_temp = pd.DataFrame({})

    for antares_result in antares_results:
        if "fullyear" in antares_result:
            balmorel_result = balmorel_results[0]
            scenario = "fullyear"
        elif "ksmooth" not in antares_result:
            balmorel_result = balmorel_results[1]
            scenario = "timeslices"
        else:
            scenario = "vectorised_ksmooth"

        balmorel_ptx, antares_ptx = get_ptx_results(
            antares_result, balmorel_result, gams_system_directory 
        )

        # Get metadata
        if 'ksmooth' not in antares_result:
            cluster_size = int(antares_result[antares_result.find('clsize'):].split('_')[0].replace('clsize', ''))
            iteration = int(antares_result[antares_result.find('nr'):].split('_')[0].replace('nr', ''))
        else:
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
        "Workflow/OverallResults/PtX_demand_comparison_clustersize.csv"
    )

@CLI.command()
def eutests():
    """
    The first tests with kernel smoothing on European scale
    """

    antbalm_list = [
        ["MainResults_EUtest_S4T56_Iter0.gdx", "20250902-2142eco-eutest_s4t56_2_5pctbw_iter0_y-2050"],
        ["MainResults_EUtest_S4T56_Iter0.gdx", "20250901-1609eco-eutest_s4t56_iter0_y-2050"],
        ["MainResults_EUtest_S4T168_Iter0.gdx","20250901-1639eco-eutest_s4t168_iter0_y-2050"],
        ["MainResults_EUFictDem_S4T56_Iter0.gdx", "20250902-1913eco-eufictdem_s4t56_2_5pctbw_iter0_y-2050"],
        ["MainResults_EUFictDem_S4T56_Iter0.gdx","20250830-1817eco-eufictdem_s4t56_iter0_y-2050" ],
    ]

    collect_ptx_results(antbalm_list, 'EUtests')

@CLI.command()
def ens_cost_sens():

    antbalm_list = [["MainResults_baf_test_new_Iter0.gdx", "20250904-1906eco-baf_test_new_ksmooth_highxcap_iter0_y-2050"],
        ["MainResults_baf_test_new_Iter0.gdx", "20250904-2148eco-baf_test_new_ksmooth_highxcap_allequalenscost_iter0_y-2050"], 
        ["MainResults_baf_test_new_Iter0.gdx", "20250905-0945eco-baf_test_new_ksmooth_hxcap_eqensc_nothermscbuild_iter0_y-2050"],
        ["MainResults_baf_test_new_Iter0.gdx", "20250909-1123eco-baf_test_fullyear_ksmooth_area_capexincluded_glmima_loensc_05bw_mingen3x_iter0_y-2050"],
    ]

    collect_ptx_results(antbalm_list, 'ens_cost_sens')

@CLI.command()
def globalminmax_dr():
    antbalm_list = [
        ['MainResults_baf_test_fullyear_Iter0.gdx','20250908-1602eco-baf_test_fullyear_ksmooth_area_capexincluded_alwaysens_iter0_y-2050'],
        ['MainResults_baf_test_fullyear_Iter0.gdx','20250908-1710eco-baf_test_fullyear_ksmooth_area_capexincluded_globalminmax_iter0_y-2050'],
        ['MainResults_baf_test_fullyear_Iter0.gdx','20250908-1805eco-w52t24_dist_wy2000_ksmooth_area_capexincluded_globalminmax_lowerenscost_iter0_y-2050'],
        ['MainResults_baf_test_fullyear_Iter0.gdx','20250909-1550eco-baf_test_fullyear_ksmooth_area_capexincluded_glmima_loensc_05bw_mingen3x_fixedbatweek_iter0_y-2050']
    ]

    collect_ptx_results(antbalm_list, 'globalminmax_dr')

@CLI.command()
def temporal_sens():

    antbalm_list = [
        ["MainResults_baf_test_fullyear_Iter0.gdx", "20250907-1114eco-baf_test_fullyear_ksmooth_bw10pct_iter0_y-2050"],
        [["MainResults_W52T24_dist_WY2000_Iter0.gdx", 'W52T24_dist_WY2000'], "20250907-1236eco-w52t24_dist_wy2000_ksmooth_bw10pct_iter0_y-2050"],
        ["MainResults_baf_test_new_Iter0.gdx", "20250905-0945eco-baf_test_new_ksmooth_hxcap_eqensc_nothermscbuild_iter0_y-2050"]
    ]

    collect_ptx_results(antbalm_list, 'temporal-sens')

@CLI.command()
def bandwidth():
 
    antbalm_list = [
        ["MainResults_baf_test_fullyear_Iter0.gdx", "20250627-1815eco-baf_test_new_fullyear_clsize1000_nr1_y-2050"],
        ["MainResults_baf_test_fullyear_Iter0.gdx", "20250907-1034eco-baf_test_fullyear_ksmooth_overfit_iter0_y-2050"],
        ["MainResults_baf_test_fullyear_Iter0.gdx", "20250907-1114eco-baf_test_fullyear_ksmooth_bw10pct_iter0_y-2050"],
        ["MainResults_baf_test_fullyear_Iter0.gdx", "20250907-2256eco-baf_test_fullyear_ksmooth_area_iter0_y-2050"],
    ]

    collect_ptx_results(antbalm_list, 'bandwidth')

@CLI.command()
def virginie_clustering():
    antbalm_list = [
        [["MainResults_noh_fullyear_Iter0.gdx"  , "noh"], "20250919-1524eco-noh_fullyear_fullyear_cl4_iter0_y-2050"],
        [["MainResults_noh_fullyear_Iter0.gdx"  , "noh"], "20250919-1528eco-noh_fullyear_fullyear_cl8_iter0_y-2050"],
        [["MainResults_noh_fullyear_Iter0.gdx"  , "noh"], "20250919-1532eco-noh_fullyear_fullyear_cl52_iter0_y-2050"],
        [["MainResults_noh_fullyear_Iter0.gdx"  , "noh"], "20250919-1536eco-noh_fullyear_fullyear_cl168_iter0_y-2050"],
        [["MainResults_noh_fullyear_Iter0.gdx"  , "noh"], "20250921-2028eco-noh_fullyear_h2vrehexo_cl672_iter0_y-2050"],
        [["MainResults_noh_fullyear_Iter0.gdx"  , "noh"], "20250921-2034eco-noh_fullyear_h2vrehexo_cl1344_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250914-0829eco-noh2_fullyear_wdisloss_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250914-0858eco-noh2_fullyear_wdisloss_wcapex_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250914-1516eco-noh2_fullyear_cl4_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250914-1520eco-noh2_fullyear_cl8_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250914-1526eco-noh2_fullyear_cl52_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250914-1532eco-noh2_fullyear_cl168_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250914-1542eco-noh2_fullyear_cl672_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250914-1556eco-noh2_fullyear_cl1344_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250915-1434eco-noh2_fullyear_cl4_fixedfuelprice_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250915-1345eco-noh2_fullyear_cl8_fixedfuelprice_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250915-1439eco-noh2_fullyear_cl52_fixedfuelprice_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250915-1445eco-noh2_fullyear_cl168_fixedfuelprice_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250915-1456eco-noh2_fullyear_cl672_fixedfuelprice_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250915-1510eco-noh2_fullyear_cl1344_fixedfuelprice_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], ""],
        # [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250914-0029eco-h2_fullyear_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250914-0844eco-h2_fullyear_wdisloss_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250914-0914eco-h2_fullyear_wdisloss_wcapex_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250914-1601eco-h2_fullyear_cl4_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250914-1606eco-h2_fullyear_cl8_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250914-1612eco-h2_fullyear_cl52_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250914-1620eco-h2_fullyear_cl168_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250914-1633eco-h2_fullyear_cl672_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250914-1651eco-h2_fullyear_cl1344_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx"  , "h2"], "20250915-1514eco-h2_fullyear_cl4_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx"  , "h2"], "20250915-1350eco-h2_fullyear_cl8_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx"  , "h2"], "20250915-1520eco-h2_fullyear_cl52_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx"  , "h2"], "20250915-1529eco-h2_fullyear_cl168_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx"  , "h2"], "20250915-1542eco-h2_fullyear_cl672_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx"  , "h2"], "20250915-1600eco-h2_fullyear_cl1344_fixedfuelprice_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"  , "h2"], "20250915-1405eco-h2_fullyear_cl8_fixedfuelprice_nocapex_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"  , "h2"], ""],
        [["MainResults_h2_fullyear_Iter0.gdx"    , "h2"], "20250915-1350eco-h2_fullyear_cl8_fixedfuelprice_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx", "h2_lss"], "20250914-1227eco-h2_lss_fullyear_wdisloss_wcapex_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx"    , "h2_lss"], "20250914-1656eco-h2_lss_fullyear_cl4_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx"    , "h2_lss"], "20250914-1700eco-h2_lss_fullyear_cl8_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx"    , "h2_lss"], "20250914-1705eco-h2_lss_fullyear_cl52_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx"    , "h2_lss"], "20250914-1712eco-h2_lss_fullyear_cl168_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx"    , "h2_lss"], "20250914-1722eco-h2_lss_fullyear_cl672_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx"    , "h2_lss"], "20250914-1735eco-h2_lss_fullyear_cl1344_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx"  , "h2_lss"], "20250915-1409eco-h2_lss_fullyear_cl8_fixedfuelprice_nocapex_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx"  , "h2_lss"], "20250915-1604eco-h2_lss_fullyear_cl4_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx"  , "h2_lss"], "20250915-1354eco-h2_lss_fullyear_cl8_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx"  , "h2_lss"], "20250915-1610eco-h2_lss_fullyear_cl52_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx"  , "h2_lss"], "20250915-1616eco-h2_lss_fullyear_cl168_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx"  , "h2_lss"], "20250915-1626eco-h2_lss_fullyear_cl672_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx"  , "h2_lss"], "20250915-1639eco-h2_lss_fullyear_cl1344_fixedfuelprice_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1236eco-h2_lss_h2t_fullyear_h2vrehexo_cl4_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1240eco-h2_lss_h2t_fullyear_h2vrehexo_cl8_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1246eco-h2_lss_h2t_fullyear_h2vrehexo_cl52_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1253eco-h2_lss_h2t_fullyear_h2vrehexo_cl168_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1304eco-h2_lss_h2t_fullyear_h2vrehexo_cl672_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1319eco-h2_lss_h2t_fullyear_h2vrehexo_cl1344_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx"  , "h2_lss"], ""],
        # ["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"],
    ]

    collect_ptx_results(antbalm_list, 'virginie_clustering')
 
@CLI.command()
def virginie_data():
    antbalm_list = [
        [["MainResults_noh_fullyear_Iter0.gdx", "noh"], "20250919-1558eco-noh_fullyear_h2surhsur_oldrounding_iter0_y-2050"],
        [["MainResults_noh_fullyear_Iter0.gdx", "noh"], "20250919-1554eco-noh_fullyear_h2vrehsur_oldrounding_iter0_y-2050"],
        [["MainResults_noh_fullyear_Iter0.gdx", "noh"], "20250919-1549eco-noh_fullyear_h2surhexo_oldrounding_iter0_y-2050"],
        [["MainResults_noh_fullyear_Iter0.gdx", "noh"], "20250919-1545eco-noh_fullyear_h2vrehexo_oldrounding_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx", "noh2"], "20250919-1804eco-noh2_fullyear_h2surhsur_oldrounding_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx", "noh2"], "20250919-1756eco-noh2_fullyear_h2vrehsur_oldrounding_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx", "noh2"], "20250919-1750eco-noh2_fullyear_h2surhexo_oldrounding_iter0_y-2050"],
        [["MainResults_noh2_fullyear_Iter0.gdx", "noh2"], "20250919-1743eco-noh2_fullyear_h2vrehexo_oldrounding_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx", "h2_lss"], "20250919-1843eco-h2_lss_fullyear_h2vrehexo_oldrounding_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx", "h2_lss"], "20250919-1850eco-h2_lss_fullyear_h2surhexo_oldrounding_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx", "h2_lss"], "20250919-1857eco-h2_lss_fullyear_h2vrehsur_oldrounding_iter0_y-2050"],
        [["MainResults_h2_lss_fullyear_Iter0.gdx", "h2_lss"], "20250919-1904eco-h2_lss_fullyear_h2surhsur_oldrounding_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx", "h2"], "20250919-1812eco-h2_fullyear_h2vrehexo_oldrounding_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx", "h2"], "20250919-1820eco-h2_fullyear_h2surhexo_oldrounding_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx", "h2"], "20250919-1828eco-h2_fullyear_h2vrehsur_oldrounding_iter0_y-2050"],
        [["MainResults_h2_fullyear_Iter0.gdx", "h2"], "20250919-1837eco-h2_fullyear_h2surhsur_oldrounding_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1253eco-h2_lss_h2t_fullyear_h2vrehexo_cl168_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1327eco-h2_lss_h2t_fullyear_h2surhexo_cl168_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1334eco-h2_lss_h2t_fullyear_h2vrehsur_cl168_iter0_y-2050"],
        [["MainResults_h2_lss_h2t_fullyear_Iter0.gdx", "h2_lss_h2t"], "20250921-1342eco-h2_lss_h2t_fullyear_h2surhsur_cl168_iter0_y-2050"],
        # [["MainResults_noh2_fullyear_Iter0.gdx"  , "noh2"], "20250915-1445eco-noh2_fullyear_cl168_fixedfuelprice_iter0_y-2050"],
        # [["MainResults_h2_fullyear_Iter0.gdx"  , "h2"], "20250915-1529eco-h2_fullyear_cl168_fixedfuelprice_iter0_y-2050"],
        # [["MainResults_h2_lss_fullyear_Iter0.gdx"  , "h2_lss"], "20250915-1616eco-h2_lss_fullyear_cl168_fixedfuelprice_iter0_y-2050"],
    ]

    collect_ptx_results(antbalm_list, 'virginie_data')


@CLI.command()
def new_sensitivities():

    collection_list = []
    antares_outputs = Path('Antares/output').glob('202510*_h2*h*_cl*_iter0_y-2050')
    for antares_output in antares_outputs:
        scenario = antares_output.name.split('eco-')[1].split('_cl')[0][:-10]
        collection_list.append(
            [[f"MainResults_{scenario}_dispatch_WY2000_Iter0.gdx", scenario],
            [antares_output.name, "00019"]]
        )

    collect_ptx_results(collection_list, 'new_sensitivities')

@CLI.command()
@click.argument('clustersize', type=str, default='cl1344')
def multiweather(clustersize: str = 'cl1344'):
    """
    The small-scale multiweather year cross-runs

    Args:
       clustersize (str): Either 'cl1344' or '672' for the two runs.
    """
    
    weather_years = [1982+i for i in range(35)]
    for training_year in weather_years:
        collection_list = []
        antares_outputs = Path('Antares/output').glob(f'*wy{training_year}_{clustersize}*')
        for antares_output in antares_outputs:
            scenario = antares_output.name.split('eco-')[1].split('_wy')[0]
            for test_year in weather_years:
                collection_list.append(
                    [[f"MainResults_{scenario}_dispatch_WY{test_year}_Iter0.gdx", scenario],
                    [antares_output.name, f"{test_year-1982+1:05.0f}"]]
                )

        collect_ptx_results(collection_list, f'multiweather_{clustersize}_{training_year}trained')

@CLI.command()
def largescale():
    result_collection = [
        [["MainResults_noh_eu_rorfix_operun_Iter0.gdx", "noh"],["20251116-1300eco-noh_eu_rorfix_wy2000_1344_h2exohexo_iter0_y-2050", "00019"]],
        [["MainResults_noh2_eu_rorfix_operun_Iter0.gdx", "noh2"], ["20251112-1718eco-noh2_eu_rorfix_wy2000_1344_h2exohexo_iter0_y-2050", "00019"]],
        [["MainResults_h2_eu_rorfix_operun_Iter0.gdx", "h2"], ["20251113-0212eco-h2_eu_rorfix_wy2000_1344_h2exohexo_iter0_y-2050", "00019"]],
        [["MainResults_h2_lss_eu_rorfix_operun_Iter0.gdx", "h2_lss"], ["20251114-1145eco-h2_lss_eu_rorfix_wy2000_1344_h2exohexo_iter0_y-2050", "00019"]],
        [["MainResults_h2_lss_h2t_eu_rorfix_operun_Iter0.gdx", "h2_lss_h2t"], ["20251115-1951eco-h2_lss_h2t_eu_rorfix_wy2000_1344_h2exohexo_iter0_y-2050", "00019"]],
    ]

    collect_ptx_results(result_collection, 'largescale')

@CLI.command()
def h2exotest():
    
    result_list = [
        [["MainResults_h2_dispatch_WY1983_Iter0.gdx", "h2"], ["20250922-1419eco-h2_wy1983_cl1344_iter0_y-2050", "00002"]],
        [["MainResults_h2_dispatch_WY2016_Iter0.gdx", "h2"], ["20250922-1419eco-h2_wy1983_cl1344_iter0_y-2050", "00035"]],
        [["MainResults_h2_dispatch_WY1983_Iter0.gdx", "h2"], ["20250929-1425eco-h2_wy1983_cl1344_h2exohexo_iter0_y-2050", "00002"]],
        [["MainResults_h2_dispatch_WY2016_Iter0.gdx", "h2"], ["20250929-1425eco-h2_wy1983_cl1344_h2exohexo_iter0_y-2050", "00035"]],
    ]

    collect_ptx_results(result_list, 'multiweather_h2exotest')

@CLI.command()
def flowbased():
    
    antbalm_list = [
        [["MainResults_noh_eu_operun_flowbased_Iter0.gdx", "noh"], ["20251019-1940eco-noh_h2vrehexo_stofixflowbased_iter0_y-2050", "00019"]],
        [["MainResults_noh2_eu_operun_flowbased_Iter0.gdx", "noh2"], ["20251020-0053eco-noh2_h2exohexo_stofixflowbased_iter0_y-2050", "00019"]],
        [["MainResults_h2_eu_operun_flowbased_Iter0.gdx", "h2"], ["20251020-0813eco-h2_h2exohexo_stofixflowbased_iter0_y-2050", "00019"]],
        [["MainResults_h2_lss_eu_operun_flowbased_Iter0.gdx", "h2_lss"], ["20251020-1738eco-h2_lss_h2exohexo_stofixflowbased_iter0_y-2050", "00019"]],
        [["MainResults_h2_lss_h2t_eu_operun_flowbased_Iter0.gdx", "h2_lss_h2t"], ["20251021-0158eco-h2_lss_h2t_h2exohexo_stofixflowbased_iter0_y-2050", "00019"]],
    ]

    collect_ptx_results(antbalm_list, 'flowbasedtest')

@CLI.command()
@click.pass_context
@click.argument('balmorel_scenario', required=True)
@click.argument('scenario_folder', required=False, default='base')
def latest(ctx, balmorel_scenario: str, scenario_folder:str):
    """Get the most recent Antares output and compare to Balmorel file"""

    balmorel_ptx, antares_ptx = get_ptx_results(
        'latest',
        'MainResults_'+balmorel_scenario+'.gdx',
        ctx.obj['gams_system_directory'],
        scenario_folder    
    )

    results = pd.concat(
        (
            balmorel_ptx,
            antares_ptx
    ), ignore_index=True).sort_values(by=['Region', 'Category'])

    print(results.to_string())
    print('\nIn total:')
    print(results.pivot_table(index='Category',
                    columns='Model',
                    values='Value',
                    aggfunc='sum'),

    )

if __name__ == "__main__":
    CLI()
