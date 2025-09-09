"""
Created on 30.03.2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)

IN ONE SENTENCE:
Converts Balmorel results to Antares input

ASSUMPTIONS:
- Peak production in VRE series = Peak capacity (but 5% loss inherent in profile, see Pre-Processing.py)
- Full transmission capacity available all hours
- Hydrogen related power production (fuel cells) can't supply electricity for heat or hydrogen production

OTHER:
Read this script from the bottom and up to get an overview
"""
### ------------------------------- ###
###       0. Script Settings        ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import click
import gams
import os
import pickle
import configparser
from functools import partial
from multiprocessing import Pool
from Functions.GeneralHelperFunctions import (
    create_transmission_input,
    get_marginal_costs,
    get_efficiency,
    get_capex,
    set_cluster_attribute,
    AntaresInput,
    get_balmorel_time_and_hours,
    data_context,
    set_scenariobuilder_values,
)
from Functions.build_supply_curves import (
    get_prices_demands,
    get_supply_curve_parameters_fit,
    get_supply_curve_parameters_all,
    load_OSMOSE_data_to_context,
    model_supply_curves_in_antares,
)
from Functions.physicality_of_antares_solution import BalmorelFullTimeseries
from Functions.kernel_2Dsmoothing import do_kernel_smoothing
from pybalmorel import Balmorel, MainResults
from pybalmorel.utils import symbol_to_df


# %% ------------------------------- ###
###           1. Functions          ###
### ------------------------------- ###
def antares_vre_capacities(
    db: gams.GamsDatabase,
    B2A_ren: dict,
    A2B_regi: dict,
    GDATA: pd.DataFrame,
    ANNUITYCG: pd.DataFrame,
    fAntTechno: pd.DataFrame,
    i: int,
    year: str,
):
    """Antares renewable capacities

    Args:
        db (gams.GamsDatabase): _description_
        B2A_ren (dict): _description_
        A2B_regi (dict): _description_
        GDATA (pd.DataFrame): _description_
        ANNUITYCG (pd.DataFrame): _description_
        fAntTechno (pd.DataFrame): _description_
        i (int): _description_
        year (str): _description_
    """

    print("\nVRE capacities to Antares...\n")

    # 1.2 Capacities to dataframe
    cap = symbol_to_df(
        db,
        "G_CAP_YCRAF",
        ["Y", "C", "R", "A", "G", "F", "Commodity", "Tech", "Var", "Unit", "Value"],
    )

    for tech in B2A_ren.keys():
        # Filter tech
        idx = (cap["Tech"] == tech) & (cap["Y"] == str(year))

        p = "../Antares/input/%s/series/" % (B2A_ren[tech])

        # Iterate through Antares areas
        for region in A2B_regi.keys():
            # Read Antares Config file for region
            area_config = configparser.ConfigParser()
            area_config.read(
                "Antares/input/renewables/clusters/%s/list.ini" % region.lower()
            )

            # Sum capacity from Balmorel Regions
            tech_cap = 0

            # If Balmorel has higher spatial resolution...
            if len(A2B_regi[region]) > 1:
                for balmorel_region in A2B_regi[region]:
                    tech_cap += (
                        cap.loc[idx & (cap.R == balmorel_region), "Value"].sum() * 1000
                    )
            # ...otherwise
            else:
                idx_cap = idx & (cap.R == region)
                tech_cap = cap.loc[idx_cap, "Value"].sum() * 1000
                capex = get_capex(cap, idx_cap, GDATA, ANNUITYCG)

            # Raise error if section doesn't exist but Balmorel invested in capacity
            if B2A_ren[tech] not in area_config.sections() and tech_cap > 1e-5:
                raise ValueError(
                    f"Balmorel invested in a {B2A_ren[tech]} capacity in {region} that isn't configured in Antares!"
                )

            if tech_cap > 1e-5:
                area_config.set(B2A_ren[tech], "nominalcapacity", str(tech_cap))
                area_config.set(B2A_ren[tech], "enabled", "true")
            else:
                if B2A_ren[tech] in area_config.sections():
                    area_config.set(B2A_ren[tech], "enabled", "false")
                    area_config.set(B2A_ren[tech], "nominalcapacity", "0")

            # Save data
            # ASSUMPTION: Peak production = 95% of Capacity (See pre-processing script)
            # ((f * tech_cap).astype(int)).to_csv(p + B2A_ren[tech] + '_%s.txt'%region, sep='\t', header=None, index=None)
            with open(
                "Antares/input/renewables/clusters/%s/list.ini" % region.lower(), "w"
            ) as configfile:
                area_config.write(configfile)
            print(region, B2A_ren[tech], round(tech_cap, 2), "MW")

            # Save technoeconomic data to file
            fAntTechno.loc[(i, year, region, tech), "CAPEX"] = capex
            fAntTechno.loc[(i, year, region, tech), "OPEX"] = 0
            fAntTechno.loc[(i, year, region, tech), "Power Capacity"] = tech_cap

    return fAntTechno, cap


def antares_thermal_capacities(
    db: gams.GamsDatabase,
    A2B_regi: dict,
    A2B_regi_h2: dict,
    BalmTechs: dict,
    GDATA: pd.DataFrame,
    FPRICE: pd.DataFrame,
    FDATA: pd.DataFrame,
    EMI_POL: pd.DataFrame,
    ANNUITYCG: pd.DataFrame,
    cap: pd.DataFrame,
    i: int,
    year: str,
    fAntTechno: pd.DataFrame,
):
    """Creates thermal capacities

    Args:
        db (gams.GamsDatabase): _description_
        A2B_regi (dict): _description_
        A2B_regi_h2 (dict): _description_
        BalmTechs (dict): _description_
        GDATA (pd.DataFrame): _description_
        FPRICE (pd.DataFrame): _description_
        FDATA (pd.DataFrame): _description_
        EMI_POL (pd.DataFrame): _description_
        ANNUITYCG (pd.DataFrame): _description_
        cap (pd.DataFrame): _description_
        i (int): _description_
        year (str): _description_
        fAntTechno (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """

    print("\nThermal capacities to Antares...\n")

    # Get economic parameters

    # Overall
    ant_input = AntaresInput("Antares")

    # Hourly hydrogen price
    h2_price_hourly = symbol_to_df(db, "H2_PRICE_YCRST")

    # Get production

    # Annual
    pro = symbol_to_df(
        db,
        "PRO_YCRAGF",
        ["Y", "C", "R", "A", "G", "F", "Commodity", "Tech", "Unit", "Value"],
    )

    # Hourly (Only needed for fuel cell)
    production_hourly = symbol_to_df(db, "PRO_YCRAGFST").query(
        'Technology == "FUELCELL"'
    )

    # Placeholders for modulation and data
    thermal_modulation = "\n".join(["1\t1\t1\t0" for i in range(8760)]) + "\n"
    thermal_data = "\n".join(["1\t1\t0\t0\t0\t0" for i in range(365)]) + "\n"

    # 2.1 Go through regions
    thermal_config = configparser.ConfigParser()
    for region in A2B_regi.keys():
        # 2.2 Get tech capacities
        thermal_config.read(
            "Antares/input/thermal/clusters/%s/list.ini" % region.lower()
        )

        # Technologies as defined by aggregated tech categories in BalmTechs dict
        for tech in BalmTechs.keys():
            if "CCS" in tech:
                CCStech = True
            else:
                CCStech = False

            # Fuels as defined by BalmTechs dict
            for fuel in BalmTechs[tech].keys():
                tech_cap = 0
                mc_cost = 0
                Nreg = 0  # Amount of Balmorel regions with this technology
                eff = 0  # Efficiency
                capex = 0
                for balmorel_region in A2B_regi[region]:
                    # Get weight from amount of corresponding areas in Balmorel
                    # weight = B2A_DE_weights[balmorel_region][region]
                    weight = 1

                    # Index for capacities
                    idx_cap = (
                        (cap["Commodity"] == "ELECTRICITY")
                        & (cap.R == balmorel_region)
                        & (cap.F == fuel)
                        & (cap.Tech == tech.replace("-CCS", ""))
                        & (cap.Y == year)
                    )

                    # Index for production
                    idx2 = (
                        (pro["Commodity"] == "ELECTRICITY")
                        & (pro["R"] == balmorel_region)
                        & (pro["F"] == fuel)
                        & (pro["Tech"] == tech.replace("-CCS", ""))
                        & (pro["Y"] == year)
                    )

                    # Filtering CCS techs
                    if CCStech:
                        idx_cap = idx_cap & (cap.G.str.find("CCS") != -1)
                        idx2 = idx2 & (pro.G.str.find("CCS") != -1)
                    else:
                        idx_cap = idx_cap & (cap.G.str.find("CCS") == -1)
                        idx2 = idx2 & (pro.G.str.find("CCS") == -1)

                    tech_cap += weight * cap.loc[idx_cap, "Value"].sum() * 1e3
                    # Get marginal costs of production
                    if cap.loc[idx_cap, "Value"].sum() * 1e3 > 1e-5:
                        # print(tech, fuel)
                        eff += get_efficiency(cap, idx_cap, GDATA)
                        capex += get_capex(cap, idx_cap, GDATA, ANNUITYCG)
                        # The technology existed in this region, so increment by one (used to average after)
                        Nreg += 1

                        mc_cost_temp = get_marginal_costs(
                            year,
                            cap,
                            idx_cap,
                            fuel,
                            GDATA,
                            FPRICE,
                            FDATA,
                            EMI_POL,
                            ANNUITYCG,
                            include_capital_costs=True,
                        )

                        if not (pd.isna(mc_cost_temp)):
                            mc_cost += mc_cost_temp  # Add to sum of marginal costs over Balmorel regions

                # Only enable tech if there's a real capacity (filtering away LP low value results)
                if tech_cap > 1e-5:
                    enabled = "true"

                    # Average marginal costs across Balmorel regions
                    try:
                        mc_cost = mc_cost / Nreg
                        eff = eff / Nreg
                        em_factor = BalmTechs[tech][fuel]["CO2"] / eff
                    except ZeroDivisionError:
                        em_factor = 0
                        print("This capacity was not used")

                    # No negative or zero marginal costs in Antares
                    if mc_cost <= 0:
                        mc_cost = 1

                else:
                    # print(region, tech, fuel, '\nCapacity: %0.2f MW\n'%tech_cap)
                    enabled = "false"
                    em_factor = 0

                # Create new cluster if it doesn't exist
                cluster_name = "%s_%s" % (tech.lower(), fuel.lower())
                if not (cluster_name in thermal_config.sections()):
                    # First, save previous edits
                    with open(
                        "Antares/input/thermal/clusters/%s/list.ini" % (region.lower()),
                        "w",
                    ) as f:
                        thermal_config.write(f)
                    thermal_config.clear()

                    # Then, create new cluster
                    ant_input.create_thermal(region.lower(), cluster_name, fuel.lower())

                    # Read again
                    thermal_config.read(
                        "Antares/input/thermal/clusters/%s/list.ini" % region.lower()
                    )

                # Make edits
                thermal_config.set(cluster_name, "enabled", enabled)
                thermal_config.set(
                    cluster_name, "nominalcapacity", str(round(tech_cap))
                )
                thermal_config.set(cluster_name, "co2", str(em_factor))

                # Create transmission capacity for hydrogen offtake, for fuel cell:
                if (tech == "FUELCELL") & (fuel == "HYDROGEN") & (tech_cap > 1e-5):
                    fuellcell_production_hours = (
                        production_hourly.query("Region == @region and Year == @year")
                        .pivot_table(index=["Season", "Time"], values="Value")
                        .index.unique()
                    )
                    # print('Production hours of fuelcell in %s: '%region, fuellcell_production_hours)

                    regional_h2_prices = h2_price_hourly.query(
                        "RRR == @region and Y == @year"
                    ).pivot_table(index=["SSS", "TTT"], values="Value")
                    # print('Price in those hours: ', regional_h2_prices.loc[fuellcell_production_hours])

                    h2_fuelcell_meanprice = regional_h2_prices.loc[
                        fuellcell_production_hours, "Value"
                    ].mean()
                    # increment marginal cost of fuelcell with hydrogen price at consumption hours
                    mc_cost += h2_fuelcell_meanprice

                    print(
                        "Average price of hydrogen when fuel cell is producing:",
                        round(h2_fuelcell_meanprice),
                        "eur/MWh",
                    )

                thermal_config.set(cluster_name, "marginal-cost", str(round(mc_cost)))
                thermal_config.set(cluster_name, "market-bid-cost", str(round(mc_cost)))

                # Save capacity timeseries (assuming no outage!)
                temp = pd.Series(np.ones(8760) * tech_cap).astype(int)
                if enabled == "true":
                    print(
                        region,
                        tech,
                        fuel,
                        "\nMarginal cost: %0.2f eur/MWh" % mc_cost,
                        "\nCapacity: %0.2f MW" % tech_cap,
                        "\nEfficiency: %0.2f pct\n" % (eff * 100),
                    )
                    try:
                        temp.to_csv(
                            "Antares/input/thermal/series/%s/%s_%s/series.txt"
                            % (region.lower(), tech.lower(), fuel.lower()),
                            sep="\t",
                            header=False,
                            index=False,
                        )

                    except OSError:
                        os.mkdir(
                            "Antares/input/thermal/series/%s/%s_%s"
                            % (region.lower(), tech.lower(), fuel.lower())
                        )
                        temp.to_csv(
                            "Antares/input/thermal/series/%s/%s_%s/series.txt"
                            % (region.lower(), tech.lower(), fuel.lower()),
                            sep="\t",
                            header=False,
                            index=False,
                        )

                    try:
                        with open(
                            "Antares/input/thermal/prepro/%s/%s_%s/modulation.txt"
                            % (region.lower(), tech.lower(), fuel.lower()),
                            "w",
                        ) as f:
                            f.write(thermal_modulation)
                        with open(
                            "Antares/input/thermal/prepro/%s/%s_%s/data.txt"
                            % (region.lower(), tech.lower(), fuel.lower()),
                            "w",
                        ) as f:
                            f.write(thermal_data)

                    except OSError:
                        os.mkdir(
                            "Antares/input/thermal/prepro/%s/%s_%s"
                            % (region.lower(), tech.lower(), fuel.lower())
                        )
                        with open(
                            "Antares/input/thermal/prepro/%s/%s_%s/data.txt"
                            % (region.lower(), tech.lower(), fuel.lower()),
                            "w",
                        ) as f:
                            f.write(thermal_data)
                        with open(
                            "Antares/input/thermal/prepro/%s/%s_%s/modulation.txt"
                            % (region.lower(), tech.lower(), fuel.lower()),
                            "w",
                        ) as f:
                            f.write(thermal_modulation)

                # Save technoeconomic data to file
                fAntTechno.loc[
                    (i, year, region, tech.lower() + "_" + fuel.lower()), "CAPEX"
                ] = capex
                fAntTechno.loc[
                    (i, year, region, tech.lower() + "_" + fuel.lower()), "OPEX"
                ] = mc_cost
                fAntTechno.loc[
                    (i, year, region, tech.lower() + "_" + fuel.lower()),
                    "Power Capacity",
                ] = tech_cap

        # Load constant PSP capacities and save in .ini

        with open(
            "Antares/input/thermal/clusters/%s/list.ini" % (region.lower()), "w"
        ) as f:
            thermal_config.write(f)
        thermal_config.clear()

        # 2.3 Get Electrolyser Capacity
        idx_cap = (
            (cap.Commodity == "HYDROGEN")
            & (cap.Tech == "ELECTROLYZER")
            & (cap.Y == year)
        )
        temp = cap.loc[idx_cap]

        tech_cap = 0
        eff = 0
        N_reg = 0
        for balmorel_region in A2B_regi[region]:
            # weight = B2A_DE_weights[balmorel_region][region]
            weight = 1
            tech_cap += (
                weight * temp[temp.R == balmorel_region].Value.sum() * 1e3
            )  # MW H2 out
            if temp.loc[temp.R == balmorel_region, "Value"].sum() * 1000 > 1e-6:
                eff += get_efficiency(cap, idx_cap & (cap.R == balmorel_region), GDATA)
                N_reg += 1

    return fAntTechno


def antares_storage_capacities(
    db: gams.GamsDatabase,
    A2B_regi: dict,
    cap: pd.DataFrame,
    GDATA: pd.DataFrame,
    ANNUITYCG: pd.DataFrame,
    fAntTechno: pd.DataFrame,
    i: int,
    year: str,
):
    """Creates storage capacities

    Args:
        db (gams.GamsDatabase): _description_
        A2B_regi (dict): _description_
        cap (pd.DataFrame): _description_
        GDATA (pd.DataFrame): _description_
        ANNUITYCG (pd.DataFrame): _description_
        fAntTechno (pd.DataFrame): _description_
        i (int): _description_
        year (str): _description_

    Returns:
        _type_: _description_
    """

    print("\nStorage capacities to Antares...\n")

    # 3.1 Placeholders and data
    h2_tank_list = ""
    h2_cavern_list = {}

    # Load results on energy capacity
    sto = symbol_to_df(
        db,
        "G_STO_YCRAF",
        ["Y", "C", "R", "A", "G", "F", "Commodity", "Tech", "Var", "Unit", "Value"],
    )

    # 3.2 Battery Storage
    for region in A2B_regi.keys():
        energy_cap = 0
        power_cap = 0
        capex = 0
        for balmorel_region in A2B_regi[region]:
            # Battery capacity
            energy_cap += (
                sto.query(
                    "R == @balmorel_region and Tech == 'INTRASEASONAL-ELECT-STORAGE' and G.str.contains('BAT-LITHIO-PEAK')"
                )
                .loc[:, "Value"]
                .sum()
                * 1e3
            )  # MWh
            idx_cap = (
                (cap.R == balmorel_region)
                & (cap.Tech == "INTRASEASONAL-ELECT-STORAGE")
                & (cap.G.str.find("BAT-LITHIO") != -1)
                & (cap.Y == year)
            )
            idx_sto = (
                (sto.R == balmorel_region)
                & (sto.Tech == "INTRASEASONAL-ELECT-STORAGE")
                & (sto.G.str.find("BAT-LITHIO") != -1)
                & (sto.Y == year)
            )
            # MW unloading capacity
            power_cap += cap.loc[idx_cap, "Value"].sum() * 1e3
            capex += get_capex(sto, idx_sto, GDATA, ANNUITYCG)

        if power_cap > 1e-6:
            print(
                "%s Li-Ion (Daily) Energy Capacity: <= %d MWh"
                % (region, power_cap * 168)
            )
        # Check GDATA, charge and discharge power capacities are the same
        # GDATA[(GDATA.G.str.find('BAT-LITHIO-PEAK') != -1) & ((GDATA.Par == 'GDSTOHUNLD') | (GDATA.Par == 'GDSTOHLOAD'))]

        # Daily Energy Capacity
        with open(
            "Antares/input/bindingconstraints/00_xtra_%s_bat_3_lt.txt" % region.lower(),
            "w",
        ) as f:
            for day in range(366):
                for hour in range(23):
                    f.write(str(int(energy_cap)) + "\n")
                f.write(str(int(energy_cap / 2)) + "\n")

        set_cluster_attribute(
            "z_%s_bat_1" % region.lower(), "nominalcapacity", energy_cap, "00_xtra"
        )
        set_cluster_attribute(
            "z_%s_bat_2" % region.lower(), "nominalcapacity", energy_cap, "00_xtra"
        )

        # 'Pumping' Capacity (Charge)
        set_cluster_attribute("z_bat_gen", "nominalcapacity", power_cap, region)

        create_transmission_input(
            "./", "Antares", "00_BAT_STO", region.lower(), [0, power_cap], 0
        )

        # Save technoeconomic data to file
        fAntTechno.loc[(i, year, region, "battery"), "OPEX"] = 0
        fAntTechno.loc[(i, year, region, "battery"), "CAPEX"] = capex
        fAntTechno.loc[(i, year, region, "battery"), "Energy Capacity"] = power_cap * 24
        fAntTechno.loc[(i, year, region, "battery"), "Power Capacity"] = power_cap

    return fAntTechno


def antares_transmission_capacities(
    db: gams.GamsDatabase, A2B_regi: dict, A2B_regi_h2: dict, year: str
):
    """Creates transmission capacities

    Args:
        db (gams.GamsDatabase): _description_
        A2B_regi (dict): _description_
        A2B_regi_h2 (dict): _description_
        year (str): _description_
    """

    print("\nTransmission capacities to Antares...\n")

    # 4.1 Read Balmorel Results
    trans = symbol_to_df(
        db, "X_CAP_YCR", ["Y", "C", "RE", "RI", "Var", "Units", "Value"]
    )
    trans.loc[:, "Commodity"] = "ELECTRICITY"

    # 4.2 Read All Links
    summed_trans_capacities = trans.query(f'Y == "{year}"').pivot_table(
        index="RE",
        columns="RI",
        values="Value",
        aggfunc=lambda x: np.sum(x) * 1e3,
        fill_value=0,
    )

    print("Paranthesis is capacity in opposite direction")
    # 4.3 Go through all links
    for export_region in summed_trans_capacities.index:
        for import_region in summed_trans_capacities.columns:
            if summed_trans_capacities.loc[export_region, import_region] > 1e-3:
                export_cap = summed_trans_capacities.loc[export_region, import_region]
                import_cap = summed_trans_capacities.loc[import_region, export_region]
                print(
                    f"{export_region} - {import_region} {export_cap:0.0f} MW ({import_cap:0.0f} MW)"
                )

                create_transmission_input(
                    "./",
                    "Antares",
                    export_region,
                    import_region,
                    [export_cap, import_cap],
                    0.01,
                )

                # Make sure that it will skip this connection the next time
                summed_trans_capacities.loc[export_region, import_region] = 0
                summed_trans_capacities.loc[import_region, export_region] = 0


def antares_exogenous_electricity_demand(
    electricity_profiles: pd.DataFrame,
    electricity_demand: pd.DataFrame,
    DISLOSSEL: pd.DataFrame,
    A2B_regi: dict,
    year: str,
):
    """Create exogenous electricity profiles for Antares

    Args:
        electricity_profiles (pd.DataFrame): The timeseries
        electricity_demand (pd.DataFrame): The annual demand
        DISLOSSEL (pd.DataFrame): Distribution loss
        A2B_regi (dict): Region mapping dictionary between Antares and Balmorel
        year (str): The model year
    """

    print("Annual electricity demands to Antares...\n")

    # Go through regions
    for region in A2B_regi.keys():
        profile = np.zeros(8784)  # Annual demand in Antares node
        ann_dem = 0
        flex_dem = 0  # Annual flexible demand
        for balmorel_region in A2B_regi[region]:
            # Get weather independant profiles

            profiles = electricity_profiles.query(
                'RRR == @balmorel_region and not DEUSER.str.contains("FICTIVE")'
            ).pivot_table(
                index=["SSS", "TTT"],
                columns="DEUSER",
                values="Value",
                aggfunc="sum",
                fill_value=0,
            )
            demand = electricity_demand.query(
                'RRR == @balmorel_region and YYY == @year and not DEUSER.str.contains("FICTIVE")'
            ).pivot_table(index="DEUSER", values="Value", aggfunc="sum", fill_value=0)

            profiles = profiles / profiles.sum()

            for col in profiles.columns:
                if col in demand.index:
                    profiles.loc[:, col] = (
                        profiles.loc[:, col]
                        * demand.loc[col, "Value"]
                        / (1 - DISLOSSEL.loc[balmorel_region, "Value"])
                    )
                else:
                    profiles.loc[:, col] = profiles.loc[:, col] * 0

            # Increment demand and add distribution loss
            ann_dem += profiles.sum().sum()
            profile[:8736] += profiles.sum(axis=1)

            print("Assigning to %s..." % (region))

        print(
            "Resulting annual electricity demand in %s = %0.2f TWh\n"
            % (region, ann_dem / 1e6)
        )

        # Save
        # NOTE: Maybe do as noted above instead, so: profiles * (DE from rese + other) + DE_industry/8760 + DE_datacenter/8760
        profile[8736:] = 0
        profile = profile.round().astype(int)
        pd.DataFrame({"values": profile}).to_csv(
            "Antares/input/load/series/load_%s.txt" % (region.lower()),
            sep="\t",
            header=None,
            index=None,
        )


def antares_weekly_resource_constraints(
    A2B_regi: dict,
    B2A_ren: dict,
    BalmTechs: dict,
    year: str,
    GDATA: pd.DataFrame,
    GMAXF: pd.DataFrame,
    GMAXFS: pd.DataFrame,
    CCCRRR: pd.DataFrame,
    cap: pd.DataFrame,
):
    """Calculates residual demand profiles (electricity load - VRE profile)
    and uses this normalised series to factor on annual resource availability

    Args:
        ALLENDOFMODEL (gams.GamsDatabase): _description_
        A2B_regi (dict): _description_
        B2A_ren (dict): _description_
        BalmTechs (dict): _description_
        year (str): _description_
        GDATA (pd.DataFrame): _description_
        cap (pd.DataFrame): _description_
    """

    CCCRRR["Done?"] = False

    # Load the stochastic years used
    with open("Antares/settings/generaldata.ini", "r") as f:
        Config = "".join(f.readlines())
    stochyears = [
        int(stochyear.split("\n")[0].replace(" ", "").replace("+=", ""))
        for stochyear in Config.split("playlist_year")[1:]
    ]

    Config = configparser.ConfigParser()
    for region in A2B_regi.keys():
        Config.read("Antares/input/renewables/clusters/%s/list.ini" % region.lower())

        load = pd.read_table(
            "Antares/input/load/series/load_%s.txt" % (region.lower()), header=None
        ).loc[:, 0]

        for VRE in B2A_ren.values():
            # Production series
            try:
                f = pd.read_table(
                    f"Antares/input/renewables/series/{region}/{VRE}/series.txt".format(
                        region=region.lower(), VRE=VRE
                    ),
                    header=None,
                )

                # Get capacity input
                vrecap = Config.getfloat(VRE, "nominalcapacity")

                # Calculate mean absolute production profile through stochastic years
                vre = f.loc[:, stochyears].mean(axis=1) * vrecap
                load = load - vre  # Residual load

            except EmptyDataError | configparser.NoOptionError:
                print("No profile for %s in %s" % (VRE, region))

        # Plot Residual LDC
        # fig, ax = plt.subplots()
        # x, y = doLDC(resload, 100)
        # ax.plot(np.cumsum(x), y)

        # Sum weekly residual loads
        resload_week = load.rolling(window=168).sum()
        # Only snapshots in the end of each week
        resload_week = resload_week[167::168]
        resload_week.index = [i for i in range(1, 53)]
        resload_week = (
            resload_week - resload_week.min()
        )  # Zero availability in best month
        resload_week = resload_week / resload_week.sum()  # Normalise energy

        # All fuels, except municipal waste
        fuels = [
            fuel
            for fuel in pd.DataFrame(BalmTechs).index.to_list()
            if fuel != "MUNIWASTE" and fuel != "HYDROGEN" and fuel != "NUCLEAR"
        ]

        Config.clear()
        # Read the binding constraint
        Config.read("Antares/input/bindingconstraints/bindingconstraints.ini")

        # Just any region - regions are all within a country
        R = A2B_regi[region][0]
        country = CCCRRR[CCCRRR.R.str.find(R) != -1].index[0]

        # 6.2 Set Efficiency of Generators in region, if it has a capacity
        for fuel in fuels:
            for tech in BalmTechs.keys():
                # Calculate average efficiency of all G types
                N_reg = 0
                eff = 0
                for balmorel_region in A2B_regi[region]:
                    idx_cap = (
                        (cap["Commodity"] == "ELECTRICITY")
                        & (cap.R == balmorel_region)
                        & (cap.F == fuel)
                        & (cap.Tech == tech)
                        & (cap.Y == year)
                    )
                    if cap.loc[idx_cap, "Value"].sum() * 1000 > 1e-6:
                        eff += get_efficiency(cap, idx_cap, GDATA)
                        N_reg += 1

                if N_reg > 0:
                    eff = eff / N_reg

                    generator = "{reg}.{tech}_{fuel}".format(
                        reg=region.lower(), tech=tech.lower(), fuel=fuel.lower()
                    )
                    for section in Config.sections():
                        if generator in Config.options(section):
                            # print('%s is in section %s'%(generator, section))
                            # print('Setting %s to efficiency %0.2f'%(generator, eff))
                            Config.set(section, generator, str(round(1 / eff, 2)))

            # 6.3 Calculate Weekly Fuel Limits for all fuels but Muniwaste, if not already done
            if not (CCCRRR.loc[country, "Done?"]):
                try:
                    pot = (
                        GMAXF.loc[
                            (GMAXF.F == fuel)
                            & (GMAXF.CRA == country)
                            & (GMAXF.Y == year),
                            "Value",
                        ].values[0]
                        / 3.6
                    )  # To MWh
                except IndexError:
                    pot = 0

                # Write it
                with open(
                    "Antares/input/bindingconstraints/%sres_%s.txt"
                    % (fuel.lower(), country.lower()),
                    "w",
                ) as f:
                    for week_distribution in resload_week:
                        for i in range(7):
                            if pot > 0:
                                # If there is a potential specified
                                f.write("%0.2f\t0\t0\n" % (week_distribution * pot / 7))
                            else:
                                # If there is no potential specified, put a very high limit
                                f.write("%0.2f\t0\t0\n" % (1e12))

                    # The last week
                    if pot > 0:
                        for i in range(2):
                            f.write("%0.2f\t0\t0\n" % (week_distribution * pot / 7))
                    else:
                        for i in range(2):
                            f.write("%0.2f\t0\t0\n" % (1e12))

        # 6.4 Input weekly fuel limit for muniwaste in region
        # Calculate average efficiency of all G types
        N_reg = 0
        eff = 0
        for balmorel_region in A2B_regi[region]:
            idx_cap = (
                (cap["Commodity"] == "ELECTRICITY")
                & (cap.R == balmorel_region)
                & (cap.F == "MUNIWASTE")
                & (cap.Tech == tech)
                & (cap.Y == year)
            )
            if cap.loc[idx_cap, "Value"].sum() * 1000 > 1e-6:
                eff += get_efficiency(cap, idx_cap, GDATA)
                N_reg += 1

        if N_reg > 0:
            eff = eff / N_reg

            generator = "{reg}.{tech}_muniwaste".format(
                reg=region.lower(), tech=tech.lower()
            )
            for section in Config.sections():
                if generator in Config.options(section):
                    # print('%s is in section %s'%(generator, section))
                    # print('Setting %s to efficiency %0.2f'%(generator, eff))
                    Config.set(section, generator, str(round(1 / eff, 2)))

        # Save configfile
        with open(
            "Antares/input/bindingconstraints/bindingconstraints.ini", "w"
        ) as configfile:
            Config.write(configfile)
        Config.clear()

        # Write potential
        idx = (GMAXFS.F == "MUNIWASTE") & (GMAXFS.Y == year)
        idx2 = GMAXFS.CRA != GMAXFS.CRA

        # Aggregate, in case Balmorel is higher resolved
        weight = 0
        for balmorel_region in A2B_regi[region]:
            idx2 = idx2 | (GMAXFS.CRA == balmorel_region)

            # Disaggregate, if Antares is higher resolved
            # weight += B2A_DE_weights[balmorel_region][region] / len(A2B_regi[region])
            weight += 1
        # print('%s weight: %0.2f'%(region, weight))

        pot = GMAXFS.loc[idx & idx2].groupby(by=["S"]).aggregate({"Value": "sum"})
        with open(
            "Antares/input/bindingconstraints/muniwasteres_%s.txt" % (region.lower()),
            "w",
        ) as f:
            for week in pot.index:
                pot0 = pot.loc[week, "Value"] / 3.6 * weight  # To MWh
                for i in range(7):
                    if pot0 > 0:
                        # If there is a potential specified
                        f.write("%0.2f\t0\t0\n" % (pot0 / 7))
                    else:
                        # If there is no potential specified, put a very high limit
                        f.write("%0.2f\t0\t0\n" % (1e12))

            # The last week
            if pot0 > 0:
                for i in range(2):
                    f.write("%0.2f\t0\t0\n" % (pot0 / 7))
            else:
                for i in range(2):
                    f.write("%0.2f\t0\t0\n" % (1e12))

        # Done. Don't have to do this for the next region in the same country
        CCCRRR.loc[country, "Done?"] = True


def demand_response_constraint_RHS(
    scenario: str,
    year: int,
    commodity: str,
    node: str,
    balmorel_timeseries: BalmorelFullTimeseries,
):
    """Creates the RHS for a constraint limiting the hourly supply of electricity to heat or hydrogen

    Args:
        scenario (str): _description_
        year (int): _description_
        commodity (str): _description_
        node (str): _description_
        balmorel_timeseries (BalmorelFullTimeseries): _description_

    Returns:
        _type_: _description_
    """

    demand = balmorel_timeseries.get_summed_profile(scenario, year, commodity, node)

    # Storage capacity
    storage = (
        balmorel_timeseries.results[scenario]
        .get_result("G_CAP_YCRAF")
        .rename(columns={"Region": "RRR", "Area": "AAA"})
        .query(
            f'Year == "{year}" and Commodity == "{commodity.upper()}" and {balmorel_timeseries.symbols[commodity]["node_name"]} == "{node}" and Technology in ["INTERSEASONAL-HEAT-STORAGE", "INTRASEASONAL-HEAT-STORAGE", "H2-STORAGE"]'
        )["Value"]
        .mul(1e3)
        .sum()
    )

    # Export capacity
    if commodity == "hydrogen":
        transmission = (
            balmorel_timeseries.results[scenario]
            .get_result("XH2_CAP_YCR")
            .query(f"From == '{node}'")["Value"]
            .mul(1e3)
            .sum()
        )
    else:
        transmission = 0

    return demand + storage + transmission


def create_demand_response_hourly_constraint(
    model: Balmorel, scenario: str, year: int, gams_system_directory: str
):
    balmorel_timeseries = BalmorelFullTimeseries(
        model, gams_system_directory=gams_system_directory
    )
    balmorel_timeseries.load_data(
        scenario, overwrite=False
    )  # NOTE: Change to overwrite True when you are finished testing

    # Load RRRAAA
    sc_folder = balmorel_timeseries.model.scname_to_scfolder[scenario]
    RRRAAA = symbol_to_df(balmorel_timeseries.model.input_data[sc_folder], "RRRAAA")

    # Load electricity nodes from balmorel_timeseries
    electricity_regions = balmorel_timeseries.set["electricity"]["RRR"].unique()

    # Load heat nodes
    heat_nodes = balmorel_timeseries.set["heat"]["AAA"].unique()

    # Load binding constraints
    bc_path = "Antares/input/bindingconstraints"

    # Go through  regions
    for region in electricity_regions:
        heat_nodes_in_region = [
            node
            for node in heat_nodes
            if node in RRRAAA.query("RRR == @region")["AAA"].unique()
        ]

        heat_RHS = np.zeros(8736)
        for node in heat_nodes_in_region:
            heat_RHS += demand_response_constraint_RHS(
                scenario, year, "heat", node, balmorel_timeseries
            )

        with open("/".join([bc_path, f"{region.lower()}_heat_lt.txt"]), "w") as f:
            f.write("\n".join([str(n) for n in heat_RHS]))
            f.write("\n".join(["0" for i in range(49)]))

        hydrogen_RHS = demand_response_constraint_RHS(
            scenario, year, "hydrogen", region, balmorel_timeseries
        )

        with open("/".join([bc_path, f"{region.lower()}_hydrogen_lt.txt"]), "w") as f:
            f.write("\n".join([str(n) for n in hydrogen_RHS]))
            f.write("\n".join(["0" for i in range(49)]))


def create_demand_response(
    weather_years: list,
    result: MainResults,
    scenario: str,
    year: int,
    temporal_resolution: dict,
    parameter_x: str,
    parameter_y: str,
    area_region_relation: pd.DataFrame,
    style: str = "report",
):
    """Create demand response curves for all hours per season

    Args:
        result (MainResults): The MainResults class
        scenario (str): Scenario
        year (int): Model year
        temporal_resolution (dict): The temporal resolution of Balmorel and corresponding input data time
        parameter_x (str): The x parameter to make create supply curves from
        parameter_y (str): The y parameter to make create supply curves from
        gams_system_directory (str, optional): Directory of GAMS binary. Defaults to None.
    """

    prices_demands = {}
    antares_input = AntaresInput("Antares")
    commodities = ["HEAT", "HYDROGEN"]

    fuel_consumption = result.get_result("F_CONS_YCRAST")
    el_prices = result.get_result("EL_PRICE_YCRST")

    unserved_energy_cost = configparser.ConfigParser()
    unserved_energy_cost.read("Antares/input/thermal/areas.ini")

    for commodity in commodities:
        # Get all input parameters in proper format for all weather years
        all_parameters = get_supply_curve_parameters_all(
            result, scenario, year, commodity
        )  # all, for later
        # Get parameters used for Balmorel optimisation (temporally aggregated)
        fit_parameters = get_supply_curve_parameters_fit(
            result, scenario, year, commodity, temporal_resolution
        )  # for fitting to Balmorel results
        prices_demands[commodity] = get_prices_demands(
            scenario, year, commodity, fit_parameters, fuel_consumption, el_prices, area_region_relation
        )

        del fit_parameters

        kernel_smooth_func = partial(
            kernel_smooth_area,
            commodity,
            weather_years,
            all_parameters,
            parameter_x,
            parameter_y,
            prices_demands,
            antares_input,
        )

        antares_input_func = partial(
            antares_input_region,
            weather_years,
            all_parameters, 
            parameter_x,
            parameter_y,
            antares_input,
            commodity,
        )

        del all_parameters

        # Make fits in parallel
        areas = list(prices_demands[commodity].keys())
        regional_unserved_energy_costs = [
            unserved_energy_cost.getfloat("unserverdenergycost", area_region_relation.loc[area, 'RRR'].lower())
            for area in areas
        ]
        print(
            f"Starting to batch {areas} with current unserved energy costs: {regional_unserved_energy_costs}"
        )
        regional_unserved_energy_costs, scenario_builder_values = process_in_batches(
            areas, regional_unserved_energy_costs, kernel_smooth_func, area_region_relation, antares_input_func
        )

        # Set unserved energy cost
        regions = pd.Series(regional_unserved_energy_costs.keys()).str.split('_', expand=True)[0]
        for region in regions:
            # print(
            #     f"setting unserved energy cost in {region} to {regional_unserved_energy_costs[region]}"
            # )
            # unserved_energy_cost.set(
            #     "unserverdenergycost",
            #     region,
            #     str(regional_unserved_energy_costs[region]),
            # )
            # print(
            #     f"setting unserved energy cost in {region} to 3000 €/MWh"
            # )
            if 'heat' in region.lower() or 'hydrogen' in region.lower():
                unserved_energy_cost.set(
                    "unserverdenergycost",
                    region,
                    str(1000),
                )

        for cluster in scenario_builder_values:
            # print(f"building scenario order for {cluster}")
            set_scenariobuilder_values(cluster)

    with open("Antares/input/thermal/areas.ini", "w") as f:
        unserved_energy_cost.write(f)


def process_in_batches(
    areas, regional_unserved_energy_costs, kernel_smooth_func, area_region_relation, antares_input_func, batch_size=33
):
    """Process regions in smaller batches to control memory usage"""
    all_unserved_costs = {}
    all_scenario_values = []

    # Kernel smoothing for all areas in parallel
    with Pool() as pool:
        batch_results = pool.starmap(kernel_smooth_func, list(zip(areas)))

    # Create demand response from kernel smoothed planes in parallel batches
    regions = area_region_relation.loc[areas, 'RRR'].unique()
    for i in range(0, len(regions), batch_size):
 
        print(
            f"Processing batch {i // batch_size + 1}/{(len(areas) - 1) // batch_size + 1}"
        )

        batch_regions = regions[i : i + batch_size]
        batch_unserved_costs = regional_unserved_energy_costs[i : i + batch_size]

        # Sum kernel smoothed planes to region level
        batch_data = []
        for region, unserved_cost in zip(batch_regions, batch_unserved_costs):

            for result in batch_results:
                # Get area result
                area = next(iter(result))
                
                # Skip result if not related to region
                if area_region_relation.loc[area, 'RRR'] != region:
                    continue

                # Initiate z_cap, z_price, x0 and y0 if not in locals()
                if 'z_capacity' not in locals():
                    z_capacity, z_price, x0, y0, max_cap, min_cap = result[area]
                   
                # Otherwise, add to previous. Might not work since results are missing and not zero in GAMS output
                else:
                    z_capacity += result[area][0]
                    # z_price += result[area][1]
                    # x0 += result[area][2]
                    # y0 += result[area][3]
                    max_cap += result[area][4]
                    min_cap += result[area][5]

                print(f'Size of {area} result:')
                print(f'z_capacity {len(z_capacity)}\n', z_capacity)
                print(f'z_price {len(z_price)}\n', z_price)
                print(f'x0 {len(x0)}\n', x0)
                print(f'y0 {len(y0)}\n', y0)
                print('max_cap\n', max_cap)
                print('min_cap\n', min_cap)

            # Append to batch_data and delete z_price so it will be re-initiated
            batch_data.append((unserved_cost, region, z_capacity, z_price, x0, y0, max_cap, min_cap))
            del z_capacity


        with Pool() as pool:
            batch_results = pool.starmap(antares_input_func, batch_data)

        # Process batch results
        for unserved_cost, scenario_vals in batch_results:
            all_unserved_costs = all_unserved_costs | unserved_cost
            all_scenario_values = all_scenario_values + scenario_vals

        # Clear batch results from memory
        del batch_results

    return all_unserved_costs, all_scenario_values


def kernel_smooth_area(
    commodity: str,
    weather_years: list,
    all_parameters: pd.DataFrame,
    parameter_x: str,
    parameter_y: str,
    prices_demands: dict,
    antares_input: AntaresInput,
    area: str,
    ):
    # Do kernel smoothing
    print(
        f"Kernel smoothing {parameter_x} and {parameter_y} for {commodity} in {area}"
    )

    max_cap = prices_demands[commodity][area]['capacity'].max()
    min_cap = prices_demands[commodity][area]['capacity'].min()
    print(f'Max and min for {area}: ', max_cap, min_cap, 'MW')

    z_capacity, x0, y0 = do_kernel_smoothing(
        prices_demands[commodity][area],
        parameter_x,
        parameter_y,
        "capacity",
        0.005,
        0.005,
        plot=True,
        plot_name=f"ksmooth_{commodity}_{area}.png",
    )
    z_price, x1, y1 = do_kernel_smoothing(
        prices_demands[commodity][area],
        parameter_x,
        parameter_y,
        "price",
        0.005,
        0.005,
        plot=True,
        plot_name=f"ksmooth_{commodity}_{area}.png",
    )

    if not (np.all(x0 == x1) and np.all(y0 == y1)):
        raise ValueError("x and y were not similar from kernel smoothing output!")

    return {area : (z_capacity, z_price, x0, y0, max_cap, min_cap)}

def antares_input_region(weather_years, all_parameters, parameter_x, parameter_y,
                       antares_input, commodity, unserved_energy_cost_region, region, z_capacity, z_price, x0, y0,
                         max_cap, min_cap):

    # Create demand response
    unserved_energy_cost, scenario_builder_values = model_supply_curves_in_antares(
        weather_years,
        all_parameters,
        parameter_x,
        parameter_y,
        z_capacity,
        z_price,
        x0,
        y0,
        antares_input,
        commodity,
        region,
        unserved_energy_cost_region,
        max_cap,
        min_cap
    )

    return unserved_energy_cost, scenario_builder_values


### ------------------------------- ###
###         2. Main Function        ###
### ------------------------------- ###


@click.pass_context
def main(ctx, sc_name: str, year: str):
    """The processing of results from Balmorel to Antares

    Args:
        sc_name (str): Scenario name
        year (str): Model year
    """
    print("\n|--------------------------------------------------|")
    print("              PERI-PROCESSING")
    print("|--------------------------------------------------|\n")

    # Metadata
    if sc_name == None:
        # Otherwise, read config from top level
        print("Reading SC from Config.ini..")
        Config = configparser.ConfigParser()
        Config.read("Config.ini")
        sc_name = Config.get("RunMetaData", "SC")

    # Configuration file
    config_file_path = "Workflow/MetaResults/%s_meta.ini" % sc_name
    if not (os.path.exists(config_file_path)):
        raise FileNotFoundError("Couldnt find %s" % config_file_path)

    Config = configparser.ConfigParser()
    Config.read(config_file_path)
    SC_folder = Config.get("RunMetaData", "SC_Folder")
    gams_system_directory = Config.get("RunMetaData", "gams_system_directory")

    # Kernel smoothing parameters
    parameter_x = Config.get("PeriProcessing", "supply_curve_parameter_x")
    parameter_y = Config.get("PeriProcessing", "supply_curve_parameter_y")

    # Plot settings
    style = Config.get("Analysis", "plot_style")
    if style == "report":
        plt.style.use("default")
        fc = "white"
    elif style == "ppt":
        plt.style.use("dark_background")
        fc = "none"

    # Iteration Data
    i = Config.getint("RunMetaData", "CurrentIter")

    # Scenario
    SC = sc_name + "_Iter%d" % i

    # Context Data
    ctx.ensure_object(dict)
    ctx.obj["balmorel_weather_year"] = Config.getint(
        "PreProcessing", "balmorel_weather_year"
    )
    ctx.obj["weather_years"] = [
        1982,
        1983,
        1984,
        1985,
        1986,
        1987,
        1988,
        1989,
        1990,
        1991,
        1992,
        1993,
        1994,
        1995,
        1996,
        1997,
        1998,
        1999,
        2000,
        2001,
        2002,
        2003,
        2004,
        2005,
        2006,
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
    ]
    data_context()
    load_OSMOSE_data_to_context()

    # Dictionaries for Balmorel/Antares set translation

    # Technologies transfered from Balmorel, with marginal costs
    with open("Pre-Processing/Output/BalmTechs.pkl", "rb") as f:
        BalmTechs = pickle.load(f)

    with open("Workflow/OverallResults/%s_AT.pkl" % sc_name, "rb") as f:
        fAntTechno = pickle.load(f)

    # Renewable name mappings
    B2A_ren = {
        "SOLAR-PV": "photovoltaics",
        "WIND-ON": "onshore",
        "WIND-OFF": "offshore",
    }

    # Region mappings
    with open("Pre-Processing/Output/A2B_regi.pkl", "rb") as f:
        A2B_regi = pickle.load(f)

    with open("Pre-Processing/Output/A2B_regi_h2.pkl", "rb") as f:
        A2B_regi_h2 = pickle.load(f)

    # Load results and data

    # All input data (should have been loaded in initialisation)
    m = Balmorel("Balmorel", gams_system_directory=gams_system_directory)
    m.load_incfiles(SC_folder)
    electricity_demand = symbol_to_df(m.input_data[SC_folder], "DE")
    electricity_profiles = symbol_to_df(m.input_data[SC_folder], "DE_VAR_T")
    ctx.obj['input_data'] = m.input_data[SC_folder]
    ctx.obj['electricity_profiles'] = electricity_profiles

    # Input data from this scenario (Initialisation.py will overwrite scenario_input_data.gdx when a new master run is initiated)
    GDATA = (
        symbol_to_df(m.input_data[SC_folder], "GDATA", ["G", "Par", "Value"])
        .groupby(by=["G", "Par"])
        .aggregate({"Value": "sum"})
    )
    FDATA = (
        symbol_to_df(m.input_data[SC_folder], "FDATA", ["F", "Type", "Value"])
        .groupby(by=["F", "Type"])
        .aggregate({"Value": "sum"})
    )
    FPRICE = (
        symbol_to_df(m.input_data[SC_folder], "FUELPRICE1", ["Y", "R", "F", "Value"])
        .groupby(by=["Y", "R", "F"])
        .aggregate({"Value": "sum"})
    )
    EMI_POL = (
        symbol_to_df(
            m.input_data[SC_folder], "EMI_POL", ["Y", "C", "Group", "Par", "Value"]
        )
        .groupby(by=["Y", "C", "Group", "Par"])
        .aggregate({"Value": "sum"})
    )
    ANNUITYCG = (
        symbol_to_df(m.input_data[SC_folder], "ANNUITYCG", ["C", "G", "Value"])
        .groupby(by=["C", "G"])
        .aggregate({"Value": "sum"})
    )
    DISLOSSEL = symbol_to_df(
        m.input_data[SC_folder], "DISLOSS_E", ["R", "Value"]
    ).pivot_table(index="R", values="Value")
    GMAXF = symbol_to_df(m.input_data[SC_folder], "GMAXF", ["Y", "CRA", "F", "Value"])
    GMAXFS = symbol_to_df(
        m.input_data[SC_folder], "GMAXFS", ["Y", "CRA", "F", "S", "Value"]
    )
    RRRAAA = (
        symbol_to_df(m.input_data[SC_folder], 'RRRAAA')
        .pivot_table(index='AAA', values='RRR', aggfunc='sum')
    )
    CCCRRR = (
        symbol_to_df(m.input_data[SC_folder], 'CCCRRR', cols=['C', 'R'])
        .pivot_table(index='C', values='R', aggfunc=lambda x : ", ".join(x))
    )

    # Loading MainResults
    print(
        "Loading results for year %s from Balmorel/%s/model/MainResults_%s.gdx\n"
        % (year, SC_folder, SC)
    )
    res = MainResults(
        files="MainResults_%s.gdx" % SC,
        paths="Balmorel/%s/model/" % SC_folder,
        system_directory=gams_system_directory,
    )

    # Temporal resolution
    balmorel_index, hour_index = get_balmorel_time_and_hours(res)
    temporal_resolution = {"balmorel_index": balmorel_index, "hour_index": hour_index}
    ctx.obj['S_all'] = ['S0%d'%i for i in range(1, 10)] + ['S%d'%i for i in range(10, 53)]
    ctx.obj['T_all'] = ['T00%d'%i for i in range(1, 10)] + ['T0%d'%i for i in range(10, 100)] + ['T%d'%i for i in range(100, 169)]
    ctx.obj['ST_all'] = pd.MultiIndex.from_product((ctx.obj['S_all'], ctx.obj['T_all']))

    # Renewable Capacities
    fAntTechno, cap = antares_vre_capacities(
        res.db[SC], B2A_ren, A2B_regi, GDATA, ANNUITYCG, fAntTechno, i, year
    )

    # Thermal Capacities
    fAntTechno = antares_thermal_capacities(
        res.db[SC],
        A2B_regi,
        A2B_regi_h2,
        BalmTechs,
        GDATA,
        FPRICE,
        FDATA,
        EMI_POL,
        ANNUITYCG,
        cap,
        i,
        year,
        fAntTechno,
    )

    # Storage Capacities
    fAntTechno = antares_storage_capacities(
        res.db[SC], A2B_regi, cap, GDATA, ANNUITYCG, fAntTechno, i, year
    )

    # Transmission Capacities
    antares_transmission_capacities(res.db[SC], A2B_regi, A2B_regi_h2, year)

    # Exogenous Electricity Demand Profile
    antares_exogenous_electricity_demand(
        electricity_profiles, electricity_demand, DISLOSSEL, A2B_regi, year
    )

    # Resource Constraints
    # antares_weekly_resource_constraints(A2B_regi, B2A_ren,
    #                                     BalmTechs, year,
    #                                     GDATA, GMAXF, GMAXFS,
    #                                     CCCRRR, cap)

    # Demand response
    create_demand_response(
        ctx.obj["weather_years"],
        res,
        SC,
        int(year),
        temporal_resolution,
        parameter_x,
        parameter_y,
        RRRAAA,
        style,
    )
    # create_demand_response_hourly_constraint(m, SC, year, gams_system_directory)

    print("\n|--------------------------------------------------|")
    print("              END OF PERI-PROCESSING")
    print("|--------------------------------------------------|\n")

    # Set periprocessing_finished to true (will be set to true after peri-processing finishes)
    with open("Workflow/MetaResults/periprocessing_finished.txt", "w") as f:
        f.write("True")


@click.command()
@click.pass_context
@click.argument("scenario", type=str)
@click.argument("year", type=str)
def peri_process(ctx, scenario: str, year: str):
    try:
        main(scenario, year)

    except Exception as e:
        # If there's an error, we still want to signal that we are finished occupying the Antares compilation
        with open("Workflow/MetaResults/periprocessing_finished.txt", "w") as f:
            f.write("True")

        # Raise the error
        raise e


if __name__ == "__main__":
    peri_process()
