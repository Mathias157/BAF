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


def get_antares_choice(
    analysis: str = "weather", year: int = 2050, just_regions: bool = False,
    **kwargs
):
    scale = kwargs.get('scale')
    case = kwargs.get('case')
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

    elif analysis == "sectors" and scale == "largescale":
        el_regions = [
            "de",
            "fr",
            "es",
            "no",
            "dk",
            "fi",
            "nl",
            "se",
            "uk",
            "ee",
            "lv",
            "lt",
            "pl",
            "be",
            "it",
            "ch",
            "at",
            "cz",
            "pt",
            "sk",
            "hu",
            "si",
            "hr",
            "ro",
            "bg",
            "gr",
            "ie",
            "lu",
            "al",
            "me",
            "mk",
            "ba",
            "rs",
        ]

        hydrogen_regions = []

        if case == 0:
            antares_scenario = "20251021-1856eco-noh_h2vrehexo_stofixflowbased_iter0_y-2050"
            balmorel_scenario = "MainResults_noh_eu_operun_flowbased_Iter0.gdx"
        elif case == 1:
            antares_scenario = "20251020-0053eco-noh2_h2exohexo_stofixflowbased_iter0_y-2050"
            balmorel_scenario = "MainResults_noh2_eu_operun_flowbased_Iter0.gdx"
        elif case == 2:
            antares_scenario = "20251020-0813eco-h2_h2exohexo_stofixflowbased_iter0_y-2050"
            balmorel_scenario = "MainResults_h2_eu_operun_flowbased_Iter0.gdx"
        elif case == 3:
            antares_scenario = "20251020-1738eco-h2_lss_h2exohexo_stofixflowbased_iter0_y-2050"
            balmorel_scenario = "MainResults_h2_lss_eu_operun_flowbased_Iter0.gdx"
        elif case == 4:
            antares_scenario = "20251021-0158eco-h2_lss_h2t_h2exohexo_stofixflowbased_iter0_y-2050"
            balmorel_scenario = "MainResults_h2_lss_h2t_eu_operun_flowbased_Iter0.gdx"
        else:
            raise ValueError("Case and scale not covered!")

    elif analysis == "sectors" and scale == "smallscale":

        el_regions = [
            "de",
            "fr",
            "es",
        ]
        hydrogen_regions = []

        if case == 0:
            antares_scenario = "20250923-0522eco-noh_wy2000_cl1344_iter0_y-2050"
            balmorel_scenario = "MainResults_noh_dispatch_WY2000_Iter0.gdx"
        elif case == 1:
            antares_scenario = "20250930-1226eco-noh2_wy2000_cl1344_h2exohexo_iter0_y-2050"
            balmorel_scenario = "MainResults_noh2_dispatch_WY2000_Iter0.gdx"
        elif case == 2:
            antares_scenario = "20250930-1240eco-h2_wy2000_cl1344_h2exohexo_iter0_y-2050"
            balmorel_scenario = "MainResults_h2_dispatch_WY2000_Iter0.gdx"
        elif case == 3:
            antares_scenario = "20251008-1720eco-h2_lss_wy2000_cl1344_h2exohexo_iter0_y-2050"
            balmorel_scenario = "MainResults_h2_lss_dispatch_WY2000_Iter0.gdx"
        elif case == 4:
            antares_scenario = "20251013-0134eco-h2_lss_h2t_wy2000_cl1344_h2exohexo_iter0_y-2050"
            balmorel_scenario = "MainResults_h2_lss_h2t_dispatch_WY2000_Iter0.gdx"
        else:
            raise ValueError("Case and scale not covered!")

        # "20250930-1301eco-h2_lss_h2t_wy2000_cl1344_h2exohexo_iter0_y-2050"
        # "20250930-1215eco-noh_wy2000_cl1344_h2exohexo_iter0_y-2050"

    else:
        raise ValueError("Pick appropriate analysis")

    # Sort
    el_regions.sort()
    hydrogen_regions.sort()

    if not just_regions:
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

    else:
        return el_regions, hydrogen_regions


# ------------------------------- #
#            2. Main              #
# ------------------------------- #


@click.group()
def main():
    pass


@main.command()
@click.argument("analysis", type=str, default="weather")
@click.argument("year", type=int, default=2050)
def compare_annual_prices(analysis, year):
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

    el_prices_balm = balmorel_result.get_result("EL_PRICE_YCR")
    print(el_prices_balm.pivot_table(index="Region", values="Value"))

    if len(h2_regions) > 0:
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

@main.command()
@click.argument("analysis", type=str, default="weather")
@click.argument("scale", type=str, default="largescale")
@click.argument("year", type=int, default=2050)
def compare_weekly_prices(analysis, year, scale):

    for scale, case in [
        [scale, 0],
        [scale, 1],
        [scale, 2],
        [scale, 3],
        [scale, 4],
    ]:
        antares_result, balmorel_result, el_regions, h2_regions = get_antares_choice(
            analysis, year, **{'scale' : scale, 'case' : case}
        )
        print('Antares result:', antares_result.name)
        print('Balmorel result:', balmorel_result.sc[0])

        el_prices_ant_hourly = antares_result.collect_result_areas(
            el_regions, "MRG. PRICE", temporal="hourly"
        )
        el_prices_ant_hourly.columns = [col.upper() for col in el_prices_ant_hourly.columns]
        el_prices_ant = antares_result.collect_result_areas(
            el_regions, "MRG. PRICE", temporal="weekly"
        )
        el_prices_ant.index = [int(i)+1 for i in el_prices_ant.index]
        lold_el_ant = antares_result.collect_result_areas(
            el_regions, "LOLD", temporal="weekly"
        )
        el_prices_ant.columns = [col.upper() for col in el_prices_ant.columns]
        # print(el_prices_ant.mean(axis=1))
        # print('Total LOLD:')
        # print(lold_el_ant.sum().sum())

        # Balmorel prices
        el_prices_balm = balmorel_result.get_result("EL_PRICE_YCRST")

        # Get hourly deviations 
        el_prices_balm_hourly = el_prices_balm.pivot_table(index=['Season', 'Time'], columns='Region', values='Value')
        el_prices_balm_hourly.index = range(8736)
        # print("Mean difference across all hours:")
        # print(((el_prices_ant_hourly-el_prices_balm_hourly)).abs().mean())

        el_prices_balm = el_prices_balm.pivot_table(index="Season", columns="Region", values="Value", aggfunc='mean')
        el_prices_balm.index = [int(i.replace('S', '')) for i in el_prices_balm.index]

        print("Mean difference across all average weekly prices:")
        print(((el_prices_ant-el_prices_balm)).mean())

        fig, ax = plt.subplots()
        ax.plot(el_prices_ant.mean(axis=1), label='Antares')
        ax.plot(el_prices_balm.mean(axis=1), label='Balmorel')
        ax.legend()
        fig.savefig(f'Workflow/OverallResults/{balmorel_result.sc[0].replace('MainResults_', '').replace('.gdx','')}_{year}_elprices.png')

        if len(h2_regions) > 0:
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

@main.command()
def get_average_adequacies():
    ant = AntaresOutput(
        "20240806-2040eco-ltcapcredconsnegfeedlowinitresmar_iter0_y-2050"
    )
    ant_ade = AntaresOutput(
        "20240905-1950eco-ltcapcredconsnflirmiter6highh2ensc_iter0_y-2050"
    )

    el_regions, h2_regions = get_antares_choice("weather", 2050, just_regions=True)

    for result in [ant, ant_ade]:
        for regions in [el_regions, h2_regions]:
            adequacy = ant.collect_result_areas(regions, "LOLD", temporal="annual")
            print("Result:", result.name)
            print("Regions:", regions)
            print(adequacy.sum().mean())


if __name__ == "__main__":
    main()
