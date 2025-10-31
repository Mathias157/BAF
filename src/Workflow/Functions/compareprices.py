"""
Price comparison function

Compare prices between Antares and Balmorel

Created on 31.10.2025
@author: Mathias Berg Rosendal
         PhD Student at DTU Management (Energy Economics & Modelling)
"""
# ------------------------------- #
#        0. Script Settings       #
# ------------------------------- #

import matplotlib.pyplot as plt
import click
from pybalmorel import MainResults
from GeneralHelperFunctions import AntaresOutput
from pathlib import Path

# ------------------------------- #
#          1. Functions           #
# ------------------------------- #


def get_antares_choice(analysis: str = "weather", year: int = 2050):
    if analysis == "weather":
        el_regions = [
            "uk00",
            "ukni",
            "se04",
            "si00",
            "sk00",
            "tr00",
            "ro00",
            "rs00",
            "se01",
            "se02",
            "se03",
            "non1",
            "nos0",
            "pl00",
            "pt00",
            "me00",
            "mk00",
            "mt00",
            "nl00",
            "nom1",
            "its1",
            "itsi",
            "lt00",
            "lu00",
            "lv00",
            "ie00",
            "itca",
            "itcn",
            "itco",
            "itcs",
            "itn1",
            "itsa",
            "fi00",
            "fr00",
            "fr15",
            "gr00",
            "gr03",
            "hr00",
            "hu00",
            "dke1",
            "dkw1",
            "ee00",
            "es00",
            "bg00",
            "ch00",
            "cy00",
            "cz00",
            "de00",
            "al00",
            "at00",
            "ba00",
            "be00",
        ]
        hydrogen_regions = [
            "z_h2_c3_uk00",
            "z_h2_c3_se04",
            "z_h2_c3_si00",
            "z_h2_c3_sk00",
            "z_h2_c3_tr00",
            "z_h2_c3_ro00",
            "z_h2_c3_rs00",
            "z_h2_c3_se01",
            "z_h2_c3_se02",
            "z_h2_c3_se03",
            "z_h2_c3_non1",
            "z_h2_c3_nos0",
            "z_h2_c3_pl00",
            "z_h2_c3_pt00",
            "z_h2_c3_me00",
            "z_h2_c3_mk00",
            "z_h2_c3_mt00",
            "z_h2_c3_nl00",
            "z_h2_c3_nom1",
            "z_h2_c3_its1",
            "z_h2_c3_itsi",
            "z_h2_c3_lt00",
            "z_h2_c3_lu00",
            "z_h2_c3_lv00",
            "z_h2_c3_ie00",
            "z_h2_c3_itca",
            "z_h2_c3_itcn",
            "z_h2_c3_itcs",
            "z_h2_c3_itn1",
            "z_h2_c3_fi00",
            "z_h2_c3_fr00",
            "z_h2_c3_gr00",
            "z_h2_c3_hr00",
            "z_h2_c3_hu00",
            "z_h2_c3_dke1",
            "z_h2_c3_dkw1",
            "z_h2_c3_ee00",
            "z_h2_c3_es00",
            "z_h2_c3_bg00",
            "z_h2_c3_ch00",
            "z_h2_c3_cy00",
            "z_h2_c3_cz00",
            "z_h2_c3_de00",
            "z_h2_c3_al00",
            "z_h2_c3_at00",
            "z_h2_c3_ba00",
            "z_h2_c3_be00",
        ]
        antares_scenario = "*eco-ltcapcredconsnflirmiter6highh2ensc_iter0_y-%d" % year
        balmorel_scenario = "MainResults_LTCapCredConsNFLIRMIter6HighH2ENSC_Iter0.gdx"

    elif analysis == "sectors":
        raise ValueError("Need to define")

    else:
        raise ValueError("Pick appropriate analysis")

    # Sort
    el_regions.sort()
    hydrogen_regions.sort()

    # Find scenarios
    ## Antares
    ant = Path("Antares/output").glob(antares_scenario)
    scenarios = [scenario.name for scenario in ant]

    if len(scenarios) > 1:
        raise ValueError("Found more than one Antares scenario!")
    else:
        antares_scenario = scenarios[0]
        antares_result = AntaresOutput(antares_scenario)

    ## Balmorel
    balm = Path("Balmorel")
    scenarios = [
        scenario.name for scenario in balm.glob("./**/model/" + balmorel_scenario)
    ]
    path = [
        str(scenario.parent)
        for scenario in balm.glob("./**/model/" + balmorel_scenario)
    ]

    if len(scenarios) > 1:
        raise ValueError("Found more than one Balmorel scenario!")
    else:
        balmorel_scenario = scenarios[0]
        path = path[0]
        balmorel_result = MainResults(balmorel_scenario, path)

    return antares_result, balmorel_result, el_regions, hydrogen_regions


# ------------------------------- #
#            2. Main              #
# ------------------------------- #


@click.group()
def main():
    pass


@main.command()
@click.argument("analysis", type=str, default="weather")
@click.argument("year", type=int, default=2050)
def compare_prices(analysis, year):
    antares_result, balmorel_result, el_regions, h2_regions = get_antares_choice(
        analysis, year
    )

    el_prices_ant = antares_result.collect_result_areas(
        el_regions, "MRG. PRICE", temporal="annual"
    )
    print(el_prices_ant.T)
    lold_el_ant = antares_result.collect_result_areas(
        el_regions, "LOLD", temporal="annual"
    )
    print(lold_el_ant.T)

    # el_prices_balm = balmorel_result.get_result("EL_PRICE_YCR")
    # print(el_prices_balm.pivot_table(index="Region", values="Value"))

    h2_prices_ant = antares_result.collect_result_areas(
        h2_regions, "MRG. PRICE", temporal="annual"
    )
    print(h2_prices_ant.T)
    lold_h2_ant = antares_result.collect_result_areas(
        h2_regions, "LOLD", temporal="annual"
    )
    print(lold_h2_ant.T)

    h2_prices_balm = balmorel_result.get_result("H2_PRICE_YCR")
    print(h2_prices_balm.pivot_table(index="Region", values="Value"))


if __name__ == "__main__":
    main()
