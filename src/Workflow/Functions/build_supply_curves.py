"""
Building Supply Curves of Balmorel Results

Created on 02.05.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling) and claude.ai
"""

### ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import click
import pickle
from sklearn.cluster import KMeans
import os
import configparser
from multiprocessing import Pool
from functools import partial
from pybalmorel.utils import symbol_to_df
from pybalmorel import Balmorel, MainResults
from .GeneralHelperFunctions import (
    load_OSMOSE_data,
    create_transmission_input,
    AntaresInput,
    log_time
)

### ------------------------------- ###
###           1. Functions          ###
### ------------------------------- ###


@click.pass_context
@load_OSMOSE_data(files=["heat", "offshore_wind", "onshore_wind", "solar_pv", "load"])
def load_OSMOSE_data_to_context(ctx, data, stoch_year_data):
    """Load OSMOSE data to context

    Args:
        ctx (_type_): _description_
        data (_type_): _description_
        stoch_year_data (_type_): _description_
    """

    ctx.obj[data] = stoch_year_data

@click.pass_context
def collect_balmorel_input_data(ctx, data: str):
    db = ctx.obj['input_data']

    if data == 'load':
        inputdata = (
            ctx.obj['electricity_profiles']
            .query('DEUSER == "RESE"')
            .pivot_table(index=['SSS', 'TTT'], columns='RRR', values='Value', aggfunc='sum')
        )

    elif data == 'heat':
        inputdata = (
            symbol_to_df(db, 'DH_VAR_T')
            .query('AAA.str.contains("SPACEHEAT") and DHUSER == "RESIDENTIAL"')
            .pivot_table(index=['SSS', 'TTT'], columns='AAA', values='Value', aggfunc='sum')
        )

        inputdata.columns = inputdata.columns.str.replace('_IDVU-SPACEHEAT', '')

    elif data == 'onshore_wind':
        inputdata = (
            symbol_to_df(db, 'WND_VAR_T')
            .query('AAA.str.contains("_A")')
            .pivot_table(index=['SSS', 'TTT'], columns='AAA', values='Value', aggfunc='sum')
        )

        inputdata.columns = inputdata.columns.str.replace('_A', '')

    elif data == 'offshore_wind':
        inputdata = (
            symbol_to_df(db, 'WND_VAR_T')
            .query('AAA.str.contains("_OFF")')
            .pivot_table(index=['SSS', 'TTT'], columns='AAA', values='Value', aggfunc='sum')
        )

        inputdata.columns = inputdata.columns.str.replace('_OFF', '')

    elif data == 'solar_pv':
        inputdata = (
            symbol_to_df(db, 'SOLE_VAR_T')
            .query('AAA.str.contains("_A")')
            .pivot_table(index=['SSS', 'TTT'], columns='AAA', values='Value', aggfunc='sum')
        )

        inputdata.columns = inputdata.columns.str.replace('_A', '')

    # Make sure that zeros are included, which the GAMS API don't
    inputdata = inputdata.reindex(ctx.obj['ST_all'], fill_value=0)
    inputdata.index.names = ['Season', 'Time']
    inputdata.columns.name = 'Region'

    return inputdata

@click.pass_context
def get_inverse_residual_load(
    ctx,
    result: MainResults,
    scenario: str,
    model_year: int,
    weather_year: int,
    hour_index: list = None,
    balmorel_index: pd.MultiIndex = None,
    to_create_antares_input: bool = False,
):
    """
    Args:
        ctx (_type_): _description_
        result (MainResults): The MainResults class
        scenario (str): The scenario
        model_year (int): The model year
        weather_year (int): _description_
        hour_index (list): _description_
        balmorel_index (pd.MultiIndex): _description_
        to_create_antares_input (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: Parameters in the format expected by get_supply_curves
    """
    # Get data
    model_year = str(model_year)

    # Reduce to timeslices of Balmorel, and convert to Balmorel timeslice naming
    all_data = {}
    for data in ["onshore_wind", "offshore_wind", "solar_pv", "load", "heat"]:
        if data != "load":
            all_data[data] = ctx.obj[data][weather_year].copy()
        else:
            all_data[data] = ctx.obj[data][0].copy()

        all_data[data].index = (
            np.array(all_data[data].index) - 1
        )  # make index start from zero
        all_data[data].index.name = "time_id"

        if not (to_create_antares_input):
            all_data[data] = all_data[data].loc[hour_index]
            all_data[data].index = balmorel_index

    # Calculate VRE profiles
    capacities = (
        result.get_result("G_CAP_YCRAF")
        .query("Scenario == @scenario and Year == @model_year")
        .query('Technology in ["WIND-ON", "WIND-OFF", "SOLAR-PV"]')
        .pivot_table(
            columns=["Region"],
            index="Technology",
            values="Value",
            aggfunc="sum",
            fill_value=0,
        )
    )
    regions = capacities.columns
    all_data["onshore_wind"] = (
        all_data["onshore_wind"][regions] * capacities.loc["WIND-ON"] * 1e3
    )
    all_data["offshore_wind"] = (
        all_data["offshore_wind"][regions] * capacities.loc["WIND-OFF"] * 1e3
    )
    all_data["solar_pv"] = (
        all_data["solar_pv"][regions] * capacities.loc["SOLAR-PV"] * 1e3
    )

    # Calculate exogenous demand profiles
    el_demand = (
        result.get_result("EL_DEMAND_YCR")
        .query("Scenario == @scenario and Year == @model_year")
        .query('Category == "EXOGENOUS"')
        .pivot_table(columns=["Region"], values="Value", aggfunc="sum")
        .reindex(columns=regions, fill_value=0)
    )
    all_data["load"] = (
        all_data["load"][regions]
        / all_data["load"][regions].sum()
        * el_demand.values
        * 1e6
    )

    heat_demand = (
        result.get_result("H_DEMAND_YCRA")
        .query("Scenario == @scenario and Year == @model_year")
        .query('Category == "EXOGENOUS"')
        .pivot_table(columns=["Region"], values="Value", aggfunc="sum")
        .reindex(columns=regions, fill_value=0)
    )
    all_data["heat"] = (
        all_data["heat"][regions]
        / all_data["heat"][regions].sum()
        * heat_demand.values
        * 1e6
    )

    # Calculate inverse residual load
    inverse_residual_load = (
        all_data["onshore_wind"]
        + all_data["offshore_wind"]
        + all_data["solar_pv"]
        - all_data["load"]
        - all_data["heat"]
    )
    inverse_residual_load = (
        inverse_residual_load.stack()
        .reset_index()
        .rename(columns={"country": "Region", 0: "Value"})
    )  # format

    return inverse_residual_load


@click.pass_context
def get_exo_demand(
    ctx,
    result: MainResults,
    scenario: str,
    model_year: int,
    weather_year: int,
    hour_index: list = None,
    balmorel_index: pd.MultiIndex = None,
    to_create_antares_input: bool = False,
):
    """
    Args:
        ctx (_type_): _description_
        result (MainResults): The MainResults class
        scenario (str): The scenario
        model_year (int): The model year
        weather_year (int): _description_
        hour_index (list): _description_
        balmorel_index (pd.MultiIndex): _description_
        to_create_antares_input (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: Parameters in the format expected by get_supply_curves
    """
    # Get data
    model_year = str(model_year)

    # Reduce to timeslices of Balmorel, and convert to Balmorel timeslice naming
    all_data = {}
    for data in ["load", "heat"]:
        if data != "load":
            all_data[data] = ctx.obj[data][weather_year].copy()
        else:
            all_data[data] = ctx.obj[data][0].copy()

        all_data[data].index = (
            np.array(all_data[data].index) - 1
        )  # make index start from zero
        all_data[data].index.name = "time_id"

        if not (to_create_antares_input):
            all_data[data] = collect_balmorel_input_data(data).loc[balmorel_index]

    # Calculate exogenous demand profiles
    el_demand = (
        result.get_result("EL_DEMAND_YCR")
        .query("Scenario == @scenario and Year == @model_year")
        .query('Category == "EXOGENOUS"')
        .pivot_table(columns=["Region"], values="Value", aggfunc="sum")
    )
    regions = el_demand.columns
    all_data["load"] = (
        all_data["load"][regions]
        / all_data["load"][regions].sum()
        * el_demand.values
        * 1e6
    )

    heat_demand = (
        result.get_result("H_DEMAND_YCRA")
        .query("Scenario == @scenario and Year == @model_year")
        .query('Category == "EXOGENOUS"')
        .pivot_table(columns=["Region"], values="Value", aggfunc="sum")
        .reindex(columns=regions, fill_value=0)
    )
    all_data["heat"] = (
        all_data["heat"][regions]
        / all_data["heat"][regions].sum()
        * heat_demand.values
        * 1e6
    )

    # Calculate total exogenous load
    exo_demand = all_data["load"] + all_data["heat"]
    exo_demand = (
        exo_demand.stack()
        .reset_index()
        .rename(columns={"country": "Region", 0: "Value"})
    )  # format

    return exo_demand


@click.pass_context
def get_vre_availability(
    ctx,
    result: MainResults,
    scenario: str,
    model_year: int,
    weather_year: int,
    hour_index: list = None,
    balmorel_index: pd.MultiIndex = None,
    to_create_antares_input: bool = False,
):
    """
    Args:
        ctx (_type_): _description_
        result (MainResults): The MainResults class
        scenario (str): The scenario
        model_year (int): The model year
        weather_year (int): _description_
        hour_index (list): _description_
        balmorel_index (pd.MultiIndex): _description_
        to_create_antares_input (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: Parameters in the format expected by get_supply_curves
    """
    # Get data
    model_year = str(model_year)

    # Reduce to timeslices of Balmorel, and convert to Balmorel timeslice naming
    all_data = {}
    for data in ["onshore_wind", "offshore_wind", "solar_pv"]:
        if data != "load":
            all_data[data] = ctx.obj[data][weather_year].copy()
        else:
            all_data[data] = ctx.obj[data][0].copy()

        all_data[data].index = (
            np.array(all_data[data].index) - 1
        )  # make index start from zero
        all_data[data].index.name = "time_id"

        if not (to_create_antares_input):
            all_data[data] = collect_balmorel_input_data(data).loc[balmorel_index]

    # Calculate VRE profiles
    capacities = (
        result.get_result("G_CAP_YCRAF")
        .query("Scenario == @scenario and Year == @model_year")
        .query('Technology in ["WIND-ON", "WIND-OFF", "SOLAR-PV"]')
        .pivot_table(
            columns=["Region"],
            index="Technology",
            values="Value",
            aggfunc="sum",
            fill_value=0,
        )
    )
    regions = capacities.columns
    all_data["onshore_wind"] = (
        all_data["onshore_wind"][regions] * capacities.loc["WIND-ON"] * 1e3
    )
    offshore_regions = set(all_data["offshore_wind"].keys())
    all_data["offshore_wind"] = (
        all_data["offshore_wind"][list(set(regions) & offshore_regions)]
        * capacities.loc["WIND-OFF"]
        * 1e3
    )
    all_data["solar_pv"] = (
        all_data["solar_pv"][regions] * capacities.loc["SOLAR-PV"] * 1e3
    )

    # Calculate inverse residual load
    vre_availability = (
        all_data["onshore_wind"] + all_data["offshore_wind"] + all_data["solar_pv"]
    )
    vre_availability = (
        vre_availability.stack()
        .reset_index()
        .rename(columns={"country": "Region", 0: "Value"})
    )  # format

    return vre_availability


@click.pass_context
def get_heat_demand(
    ctx,
    result: MainResults,
    scenario: str,
    model_year: int,
    weather_year: int,
    hour_index: list = None,
    balmorel_index: pd.MultiIndex = None,
    to_create_antares_input: bool = False,
):
    """Calculate inverse residual load for the supply curve fitting functions

    Args:
        result (MainResults): The MainResults class
        scenario (str): The scenario
        year (int): The model year
        hour_index (list): The chosen hours in a year chosen as Balmorel resolution

    Returns:
        pd.DataFrame: Parameters in the format expected by get_supply_curves
    """

    # Get data
    model_year = str(model_year)
    heat_profile = ctx.obj["heat"][weather_year].copy()
    # make index start from zero
    heat_profile.index = np.array(heat_profile.index) - 1
    heat_profile.index.name = "time_id"

    if not (to_create_antares_input):
        # Reduce to timeslices of Balmorel, and convert to Balmorel timeslice naming
        heat_profile = heat_profile.loc[hour_index]
        heat_profile.index = balmorel_index

    heat_demand = (
        result.get_result("H_DEMAND_YCRA")
        .query("Scenario == @scenario and Year == @model_year")
        .query('Category == "EXOGENOUS"')
        .pivot_table(columns=["Region"], values="Value", aggfunc="sum")
    )

    # Calculate exogenous demand profiles
    heat_profile = (
        heat_profile[heat_demand.columns]
        / heat_profile[heat_demand.columns].sum()
        * heat_demand.values
        * 1e6
    )
    heat_profile = (
        heat_profile.stack()
        .reset_index()
        .rename(columns={"country": "Region", 0: "Value"})
    )

    return heat_profile


@click.pass_context
def get_supply_curve_parameters_fit(
    ctx,
    result: MainResults,
    scenario: str,
    year: int,
    commodity: str,
    temporal_resolution: dict,
):
    """Get parameters for supply curve fitting depending on the commodity

    Args:
        ctx (_type_): click CLI context
        result (MainResults): The result file
        scenario (str): The scenario name
        year (int): The model year
        commodity (str): Either 'HEAT' or 'HYDROGEN'
        temporal_resolution (dict): Temporal resolution of Balmorel

    Raises:
        ValueError: If choice is not 'HEAT' or 'HYDROGEN'

    Returns:
        parameters (tuple[pd.DataFrame]): The parameters to fit with columns [parameter_name, 'Region', 'Season', 'Time']
    """

    balmorel_weather_year = ctx.obj["balmorel_weather_year"]

    if commodity.upper() == "HEAT":
        # NOTE: SHOULD ADD SOME LINES TO MAKE THESE DEPENDENT ON THE Config/PeriProcessing/kernel_smooth_parameter_xy PARAMETERS
        exo_demand = get_exo_demand(
            result,
            scenario,
            year,
            balmorel_weather_year,
            temporal_resolution["hour_index"],
            temporal_resolution["balmorel_index"],
        )
        return exo_demand
    elif commodity.upper() == "HYDROGEN":
        # NOTE: SHOULD ADD SOME LINES TO MAKE THESE DEPENDENT ON THE Config/PeriProcessing/kernel_smooth_parameter_xy PARAMETERS
        vre_availability = get_vre_availability(
            result,
            scenario,
            year,
            balmorel_weather_year,
            temporal_resolution["hour_index"],
            temporal_resolution["balmorel_index"],
        )
        return vre_availability
    else:
        raise ValueError(
            f"Commodity '{commodity}' is not yet a part of this framework. Please choose 'HEAT' or 'HYDROGEN'"
        )

def cluster_values(group: pd.DataFrame, cluster_size: int):
    values = group['Value'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(cluster_size, len(group)), random_state=42)
    group['Cluster'] = kmeans.fit_predict(values)
    return group

@click.pass_context
def get_supply_curve_parameters_all(
    ctx, result: MainResults, scenario: str, year: int, commodity: str
):
    """Get parameters for supply curve fitting depending on the commodity

    Args:
        ctx (_type_): click CLI context
        result (MainResults): The result file
        scenario (str): The scenario name
        year (int): The model year
        commodity (str): Either 'HEAT' or 'HYDROGEN'
        temporal_resolution (dict): Temporal resolution of Balmorel

    Raises:
        ValueError: If choice is not 'HEAT' or 'HYDROGEN'

    Returns:
        parameters (pd.DataFrame): All parameters, for all weather years ['time_id', 'Region', parameter_name, 'Weather Year']
    """

    weather_years = ctx.obj["weather_years"]
    parameters = pd.DataFrame({})

    for weather_year in weather_years:
        if commodity.upper() == "HEAT":
            # NOTE: SHOULD ADD SOME LINES TO MAKE THESE DEPENDENT ON THE Config/PeriProcessing/kernel_smooth_parameter_xy PARAMETERS
            temp = get_exo_demand(
                result, scenario, year, weather_year, to_create_antares_input=True
            )
        elif commodity.upper() == "HYDROGEN":
            # NOTE: SHOULD ADD SOME LINES TO MAKE THESE DEPENDENT ON THE Config/PeriProcessing/kernel_smooth_parameter_xy PARAMETERS
            temp = get_vre_availability(
                result, scenario, year, weather_year, to_create_antares_input=True
            )
        else:
            raise ValueError(
                f"Commodity '{commodity}' is not yet a part of this framework. Please choose 'HEAT' or 'HYDROGEN'"
            )

        # Concatenate
        temp["Weather Year"] = weather_year
        parameters = pd.concat((parameters, temp))

    return parameters


def get_supply_curve(x: np.array, y: np.array):
    """A function to construct a supply curve depending on x and y data

    Args:
        x (np.array): Typically electricity prices in €/MWh
        y (np.array): Typically electricity demands in MWh

    Returns:
        fit_x, fit_y (list, list): The 'fitted' curve in x and y coordinates
    """

    df = pd.DataFrame({"x": x, "y": y})

    # Sort values by descending x
    df = df.sort_values(by="x", ascending=False)
    df.index = np.arange(len(df))

    # The 'fitted' values
    fit_x, fit_y = [], []

    for i, row in df.iterrows():
        if i == 0:
            # Fill the first values
            if small_number_to_zero(row["y"]) != 0:
                fit_x.append(small_number_to_zero(row["x"]) + 1e-3)
                fit_y.append(0)
            fit_x.append(small_number_to_zero(row["x"]))
            fit_y.append(small_number_to_zero(row["y"]))
        elif row["y"] > fit_y[-1]:
            # Add higher y at lower x as a stepwise linear function
            fit_x.append(small_number_to_zero(row["x"]) + 1e-3)
            fit_y.append(fit_y[-1])
            fit_x.append(small_number_to_zero(row["x"]))
            fit_y.append(small_number_to_zero(row["y"]))

    # Last point
    if fit_x[-1] != 0:
        fit_x.append(0)
        fit_y.append(fit_y[-1])

    return fit_x, fit_y


def small_number_to_zero(number: float):
    if number < 1e-6:
        number = 0
    return number


def seasonal_colors(num_seasons: int, style: str):
    """Create colors for all S01-SNM seasons, that are more red in the middle

    Args:
        num_seasons (int): Amount of seasons
    """

    colors = {}
    for i in range(1, num_seasons + 1):
        # Calculate how far we are from the midpoint (S26)
        # This gives a value between 0 (at S26) and 1 (at S01 or S52)
        distance_from_mid = abs(i - (num_seasons / 2 + 0.5)) / (num_seasons / 2)

        # Red component: max at midpoint, min at endpoints
        color = 1.0 - distance_from_mid
        # Black components: min at midpoint, max at endpoints
        other_color = distance_from_mid

        season_key = f"S{i:02d}"
        if style == "report":
            # Red component varies, others fixed. .to_hex used to avoid color map warning
            colors[season_key] = mcolors.to_hex([color, 0, 0, 0.5])
        else:
            # Red component varies, others fixed. .to_hex used to avoid color map warning
            colors[season_key] = mcolors.to_hex([1, other_color, other_color, 0.5])

    return colors


def find_closest_x(x0: float, xp: np.array, yp: np.array):
    try:
        x = list(xp).index(x0)
        y = yp[x]
    except ValueError:
        # x wasnt in xp
        diff = np.abs(xp - x0)
        closest_index = list(diff).index(diff.min())
        y = yp[closest_index]
    return y


def combine_step_curves(x1, y1, x2, y2):
    """
    Combine two step curves with different x-coordinates.

    Args:
        x1, y1: Coordinates of the first step curve
        x2, y2: Coordinates of the second step curve

    Returns:
        combined_x, combined_y: Coordinates of the combined step curve
    """
    # Combine and sort all unique x-coordinates
    all_x = np.unique(np.concatenate([x1, x2]))
    all_x[::-1].sort()  # Sort descending

    # Initialize arrays for combined curve
    combined_x = []
    combined_y = []

    # Evaluate the first curve at all x points
    for i, x0 in enumerate(all_x):
        if i == 0:
            combined_x.append(x0)
            combined_y.append(0)
        combined_x.append(x0)
        combined_y.append(find_closest_x(x0, x1, y1) + find_closest_x(x0, x2, y2))

    return combined_x, combined_y


def combine_multiple_supply_curves(x_list, y_list):
    """
    Combine multiple step curves with different x-coordinates.

    Args:
        x_list: List of x-coordinate arrays for each curve
        y_list: List of y-coordinate arrays for each curve

    Returns:
        combined_x, combined_y: Coordinates of the combined step curve
    """
    # Verify input
    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list must have the same length")

    if len(x_list) == 0:
        return np.array([]), np.array([])

    if len(x_list) == 1:
        return x_list[0], y_list[0]

    # Start with the first curve
    combined_x, combined_y = x_list[0], y_list[0]

    # Add each subsequent curve
    for i in range(1, len(x_list)):
        combined_x, combined_y = combine_step_curves(
            combined_x, combined_y, x_list[i], y_list[i]
        )

    return combined_x, combined_y

def get_supply_curves(scenario: str, 
                      year: int, 
                      commodity: str, 
                      parameters: pd.DataFrame,
                      fuel_consumption: pd.DataFrame, 
                      el_prices: pd.DataFrame,
                      cluster_size: int = 10,
                      plot_overall_curves: bool = False,
                      plot_all_curves: bool = False,
                      style: str = 'report'):
    """Create seasonal curves for hydrogen and heat for every region in a scenario 

    Args:
        scenario (str): Scenario to analyse
        year (int): The model year
        commodity (str): The commodity
        parameters (pd.DataFrame): The dataframe containing the parameter values for all regions, seasons and time steps with columns ['Region', 'Season', 'Time']
        fuel_consumption (pd.DataFrame): Fuel consumption results.
        el_prices (pd.DataFrame): Electricity prices.
        cluster_size (int): The amount of supply curves to build for a respective parameter (NOTE: This is problematic when you start having smaller countries in the scope! Should be made dependent on the absolute magnitude of profiles)
        plot_overall_curves (bool, optional): Plot regional heat and hydrogen supply curves?
        plot_all_curves (bool, optional): Plot stepfunction fit of ALL technologies and regional heat and hydrogen supply curves?
        style (str, optional): Style of supply curve plot. Defaults to 'report'.

    Returns:
        resulting_curves (dict): Price and capacities for all region and parameters
    """

    year = str(year)
    commodity2technology = {'HEAT' : 'ELECT-TO-HEAT', 'HYDROGEN' : 'ELECTROLYZER'}
    technology = commodity2technology[commodity]
    df1_temp = fuel_consumption.query('Year == @year and Technology == @technology')
    df2_temp = el_prices.query('Year == @year')
    
    # Convert EPS to 0
    idx = df2_temp.query('Value < 1e-6').index
    df2_temp.loc[idx, 'Value'] = 0 
    
    # Prepare parameters to iterate through and colors for plotting them
    regions = df1_temp.Region.unique()
    parameter_name = [col for col in parameters.columns if not(col in ['Region', 'Season', 'Time'])][0]
    
    ## Cluster parameters 
    parameters = parameters.groupby('Region').apply(lambda x: cluster_values(x, cluster_size))
    
    resulting_curves = {}
    for region in regions:
        resulting_curves[region] = get_curves(
            scenario,
            parameters,
            commodity,
            parameter_name,
            df1_temp,
            df2_temp,
            plot_all_curves,
            plot_overall_curves,
            region
        )

    return resulting_curves

def get_curves(scenario, parameters, commodity, parameter_name, df1_temp, df2_temp, plot_all_curves: bool, plot_overall_curves: bool, region: str):

    # Get regional parameters and amount of clusters
    resulting_curve = {}
    region_parameters = parameters.query('Region == @region')
    clusters = region_parameters['Cluster'].unique()
    
    print('='*20, '\n', f'Fitting supply curves for {commodity} in {region}...')
    print('='*20)

    parallel_func = partial(
        process_cluster,
        scenario,
        commodity,
        region,
        parameter_name,
        region_parameters,
        df1_temp,
        df2_temp,
        plot_all_curves,
        plot_overall_curves,
    )
    
    with Pool() as pool:
        batch_results = pool.starmap(parallel_func, list(zip(clusters)))

    resulting_curve = {result[0] : result[1] for result in batch_results} 

    # Plot overall curve    
    if plot_all_curves or plot_overall_curves:
        fig_season, ax_parameter = plt.subplots(facecolor='none')
        for parameter in resulting_curve.keys():
            ax_parameter.plot(resulting_curve[parameter]['price'], 
                            resulting_curve[parameter]['capacity'], 
                            )

        ax_parameter.set_title('Supply Curve for %s in %s'%(commodity, region))
        ax_parameter.set_ylabel('MWh')
        ax_parameter.set_xlabel('€/MWh')
        ax_parameter.set_facecolor('none')
        fig_season.savefig('Workflow/OverallResults/supply_curve_%s_%s.png'%(commodity, region),
                            bbox_inches='tight')

    return region, resulting_curve

def process_cluster(
        scenario: str,
        commodity: str,
        region: str,
        parameter_name: str,
        region_parameters,
        df1_temp,
        df2_temp,
        plot_all_curves,
        plot_overall_curves,
        cluster
    ):
    
    # Placeholders
    supply_curves_x, supply_curves_y = [], []
    
    # Find seasons and times of the unique parameter 
    temp = region_parameters.query('Cluster == @cluster')[['Season', 'Time', parameter_name]]
    average_parameter = np.round(temp[parameter_name].mean())
    seasons = temp['Season'].to_list()
    times = temp['Time'].to_list()
    print(f'Cluster {cluster}, Average {parameter_name} = {average_parameter}') 
    # print(f'...for seasons {seasons} and times {times}')
    
    for area in df1_temp.query('Region == @region').Area.unique():
        
        # Get technologies consuming electricity in area
        df1=df1_temp.query('Area==@area and Fuel=="ELECTRIC" and Season in @seasons and Time in @times').pivot_table(index=['Season', 'Time'], columns='Generation', values='Value', aggfunc='sum')
        
        # if len(df1.columns) > 0:
            # print(f'Electricity consumption in {area}:\n', df1)

        for tech in df1.columns:
            
            # Skip if very low fuel consumption
            if df1.loc[:, tech].max() < 1e-5:
                continue
            
            # Get electricity prices at hours
            df2=df2_temp.query('Scenario==@scenario and Region==@region and Season in @seasons and Time in @times').pivot_table(index=['Season', 'Time'], values='Value', aggfunc='mean')
            # print(f'Electricity prices:\n', df2)

            # Get combined dataframe with electricity consumption (df1) and price (df2, Value)
            temp=df1[[tech]].merge(df2[['Value']], left_index=True, right_index=True).fillna(0)
            # print(f'Consumption and price for {tech} in {area}:\n', temp)

            # Piecewise linear fit
            fit_x, fit_y = get_supply_curve(temp.loc[:, 'Value'].values.flatten(),
                                        temp.loc[:, tech].values.flatten())
            
            supply_curves_x.append(fit_x)
            supply_curves_y.append(fit_y)
                
            # Plot fit to data points for specific technology
            if plot_all_curves:
                fig, ax = plt.subplots()
                temp.plot(kind='scatter', x='Value', y=tech, ax=ax, 
                            label=average_parameter)
                ax.plot(fit_x, fit_y)
                ax.set_ylabel(f'{tech} (MWh)')
                ax.set_xlabel('Electricity Price (€/MWh)')
                ax.set_title(area)
                ax.legend(loc='center left', bbox_to_anchor=(1.05, .5))
                fig.savefig(f'Workflow/OverallResults/eldempricecurve_{commodity}_{area}_{tech}_{parameter_name}{average_parameter:0.2f}.png', bbox_inches='tight')
            
    if len(supply_curves_x) != 0:
        # print('='*10, '\nsupply curves x:\n', supply_curves_x, '\nsupply curves y:\n', supply_curves_y)
        combined_x, combined_y = combine_multiple_supply_curves(supply_curves_x, supply_curves_y)
        # print('\n', '='*10, '\ncombined supply curve x:\n', combined_x, '\ncombined supply curve y:\n', combined_y)
    else:
        combined_x, combined_y = [0, 0], [0, 0]
 
   
    # Store seasonal curves   
    resulting_curve = {'price' : np.round(combined_x),
                       'capacity' : np.round(combined_y)}

    return average_parameter, resulting_curve


def get_prices_demands(
    scenario: str,
    year: int,
    commodity: str,
    parameters: pd.DataFrame,
    fuel_consumption: pd.DataFrame,
    el_prices: pd.DataFrame,
):
    """Create seasonal curves for hydrogen and heat for every region in a scenario

    Args:
        scenario (str): Scenario to analyse
        year (int): The model year
        commodity (str): The commodity
        parameters (pd.DataFrame): The dataframe containing the parameter values for all regions, seasons and time steps with columns ['Region', 'Season', 'Time']
        fuel_consumption (pd.DataFrame): Fuel consumption results.
        el_prices (pd.DataFrame): Electricity prices.
        cluster_size (int): The amount of supply curves to build for a respective parameter (NOTE: This is problematic when you start having smaller countries in the scope! Should be made dependent on the absolute magnitude of profiles)

    Returns:
        resulting_curves (dict): Price and capacities for all region and parameters
    """

    year = str(year)
    commodity2technology = {"HEAT": "ELECT-TO-HEAT", "HYDROGEN": "ELECTROLYZER"}
    technology = commodity2technology[commodity]
    df1_temp = fuel_consumption.query(
        f'Year == "{year}" and Technology == "{technology}"'
    )
    df2_temp = el_prices.query(f'Year == "{year}"')

    # Convert EPS to 0
    idx = df2_temp.query("Value < 1e-6").index
    df2_temp.loc[idx, "Value"] = 0

    # Prepare parameters to iterate through
    regions = df1_temp.Region.unique()
    parameter_names = [
        col for col in parameters.columns if not (col in ["Region", "Season", "Time"])
    ]

    # Prepare fit result data
    output = {}

    for region in regions:
        # Get regional parameters and amount of clusters
        region_parameters = parameters.query("Region == @region")

        print(
            f"Getting electricity price and consumptions for {commodity} in {region}..."
        )

        demands = (
            df1_temp.query(f'Fuel=="ELECTRIC" and Region=="{region}"')
            .pivot_table(index=["Season", "Time"], values="Value", aggfunc="sum")
            .reset_index()
        )
        prices = (
            df2_temp.query(f'Scenario=="{scenario}" and Region=="{region}"')
            .pivot_table(index=["Season", "Time"], values="Value", aggfunc="mean")
            .reset_index()
        )
        region_parameters = region_parameters.merge(
            demands[["Season", "Time", "Value"]], on=["Season", "Time"]
        ).rename(columns={"Value": "capacity"})
        region_parameters = region_parameters.merge(
            prices[["Season", "Time", "Value"]], on=["Season", "Time"]
        ).rename(columns={"Value": "price"})

        output[region] = region_parameters.drop(columns=["Season", "Time"])

    return output


def cluster_hours(group: pd.DataFrame, cluster_size: int):
    values = group["Value"].values.reshape(-1, 1)
    # min() handles cases with <7 samples, random state is fixed
    kmeans = KMeans(n_clusters=min(cluster_size, len(group)), random_state=42)
    group["Cluster"] = kmeans.fit_predict(values)
    return group


def use_all_hours(group: pd.DataFrame):
    group["Cluster"] = np.arange(1, len(group) + 1)
    return group


def find_closest_indices_with_cut(column, Y):
    """Find the indices in a column, that are closest to each value in the list Y
    @author: claude.ai

    Args:
        column (pd.Series): A column in a dataframe, i.e. a series class
        Y (list): A list of values

    Returns:
        result: A dictionary with indices for column, for each element in Y
    """
    # Sort parameter values
    Y.sort()

    # Create bins as midpoints between consecutive Y values
    # Add -inf and +inf as boundaries
    if len(Y) == 1:
        bins = [-np.inf, np.inf]
        labels = [0]
    else:
        # Create bins at midpoints between Y values
        midpoints = (Y[:-1] + Y[1:]) / 2
        bins = [-np.inf] + midpoints.tolist() + [np.inf]
        labels = list(range(len(Y)))

    # Use pd.cut to assign each value to closest Y index
    closest_indices = pd.cut(column, bins=bins, labels=labels, include_lowest=True)

    # Group original indices by their closest Y value
    result = {}
    for y_idx in range(len(Y)):
        # print(f'Looking for parameters closest to {Y[y_idx]}, which is nr. {y_idx} of the parameters')
        mask = closest_indices == y_idx
        indices = column.index[mask].tolist()
        if indices:  # Only include if there are indices
            # print(indices)
            result[Y[y_idx]] = indices
        else:
            result[Y[y_idx]] = []
            # print(f'No indices found for param "{Y[y_idx]}"')

    return result


def map_closest_parameters(
    all_parameters: pd.DataFrame, fitted_parameters: list, region: str
):
    weather_year_array = all_parameters.query("Region == @region").pivot_table(
        index="time_id", columns="Weather Year", values="Value"
    )

    weather_years = weather_year_array.columns
    indices = {weather_year: {} for weather_year in weather_years}
    for weather_year in weather_years:
        indices[weather_year] = find_closest_indices_with_cut(
            weather_year_array[weather_year], np.array(list(fitted_parameters))
        )

    return indices

def model_supply_curves_in_antares(weather_years: list, 
                                   all_parameters: pd.DataFrame, 
                                   supply_curves: dict,
                                   antares_input: AntaresInput,
                                   commodity: str,
                                   unserved_energy_cost: configparser.ConfigParser,
                                   region: str
                                   ):
    
    print(log_time(), f'Modelling demand response in {region}')

    # Placeholder for availability, electricity to commodity load, unserved energy cost (highest marginal price + 1 €/MWh) and the parameter for all years
    availability = {}
    load = np.zeros((8760, len(weather_years)))
    highest_price = 0
    scenariobuilder_values = []
    
    # Delete all thermal clusters in virtual region
    virtual_area = f'{region}_{commodity}'.lower()

    try:
        antares_input.purge_thermal_clusters(virtual_area)
    except FileNotFoundError:
        # already not existing
        pass
    
    # Map the parameters not captured by Balmorel timeslices to the closest fitted parameter
    fitted_parameters = supply_curves[region].keys()  
    print(f'Fitting {commodity} for region {region}')  
    idx_mapped = map_closest_parameters(all_parameters, fitted_parameters, region)
    
    for parameter in fitted_parameters:
        
        # Get the supply curve for the specific parameter
        temp = pd.DataFrame({'price' : supply_curves[region][parameter]['price'],
                            'capacity' : supply_curves[region][parameter]['capacity']},
                            index=np.arange(len(supply_curves[region][parameter]['price'])))
        
        # Store max price if higher than overall highest price for region
        max_price_for_parameter = round(temp['price'].max())
        if max_price_for_parameter >= highest_price:
            highest_price = max_price_for_parameter + 1

        # Take difference between all values and sum, which will equal the availabilities at aggregated (rounded) prices
        temp['capacity'] = temp.capacity.diff().fillna(0)
        temp = temp.groupby(['price']).aggregate({'capacity' : 'sum'})
        
        # Get rounded prices
        prices = np.unique([round(price) for price in temp.index if temp.loc[price, 'capacity'] > 0])
        
        # if len(prices) > 0:
        #     print(prices)
        
        # Create a cluster per price 
        for price in prices:
            
            # Get max capacity and initiate availability timeseries if it doesn't exist yet
            cluster_name = f'{price:.0f}_europermwh'
            if not(cluster_name in availability.keys()):
                availability[cluster_name] = np.zeros((8760, len(weather_years)))
                max_cap = temp.loc[price, 'capacity']
            elif availability[cluster_name].max() > temp.loc[price, 'capacity']:
                max_cap = availability[cluster_name].max()
            else:
                max_cap = temp.loc[price, 'capacity']
            
            config, cluster_series_path, prepro_path = antares_input.create_thermal(virtual_area, cluster_name, 'lole', 
                                                                                    True, max_cap, price)
        
            # Set availability of virtual cluster
            for i, weather_year in enumerate(weather_years):
                availability[cluster_name][idx_mapped[weather_year][parameter], i] = temp.loc[price, 'capacity']
                
        # Set load
        for i, weather_year in enumerate(weather_years):
            load[idx_mapped[weather_year][parameter], i] = temp.loc[:, 'capacity'].sum()

    # Save load and availability
    np.savetxt(antares_input.path_load[virtual_area], load, delimiter='\t', fmt='%g')
        
    create_transmission_input('./', 'Antares', region, virtual_area, [load.max(), 0], 0.1)
    
    for cluster in availability.keys():
        np.savetxt(os.path.join(antares_input.path_thermal_clusters[virtual_area]['series'], cluster, 'series.txt'), 
                   availability[cluster], delimiter='\t', fmt='%g')

        scenariobuilder_values.append(
           f"t,{region.lower()}_{commodity.lower()},%d,{cluster.lower()}" 
        )
    
    # Set unserved energy cost for virtual region
    virtual_region_uec = str(highest_price)

    # Set unserved energy cost for related region higher if it is below the virtual cost
    real_region_uec = unserved_energy_cost.getfloat('unserverdenergycost', region.lower())
    if real_region_uec <= highest_price:
        real_region_uec = highest_price + 10
        
    # Set hydrogen related power production cost between the two, so fuel cells don't supply heat or hydrogen
    conf = antares_input.thermal(region)
    fuelcell_cost = conf.getfloat('fuelcell_hydrogen', 'marginal-cost')
    if fuelcell_cost <= highest_price:
        conf.set('fuelcell_hydrogen', 'marginal-cost', str(highest_price + 5))
        conf.set('fuelcell_hydrogen', 'market-bid-cost', str(highest_price + 5))
        with open('Antares/input/thermal/clusters/%s/list.ini'%region.lower(), 'w') as f:
            conf.write(f)

    return region, [virtual_region_uec, real_region_uec], scenariobuilder_values

def find_closest_indices_chunked(x, y, px_values, py_values, chunk_size=300):
    """Find closest indices using chunked processing to minimize memory usage."""
    n_px = len(px_values)
    closest_indices = np.zeros(n_px, dtype=int)
    min_distances = np.full(n_px, np.inf)

    for chunk_start in range(0, len(x), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(x))
        x_chunk = x[chunk_start:chunk_end]
        y_chunk = y[chunk_start:chunk_end]

        # Calculate distances for this chunk
        chunk_distances = np.sqrt(
            (x_chunk[:, np.newaxis] - px_values) ** 2
            + (y_chunk[:, np.newaxis] - py_values) ** 2
        )

        # Find minimum distances in this chunk
        chunk_min_distances = chunk_distances.min(axis=0)
        chunk_argmin = chunk_distances.argmin(axis=0)

        # Update global minimums where this chunk has smaller distances
        update_mask = chunk_min_distances < min_distances
        min_distances[update_mask] = chunk_min_distances[update_mask]
        # Adjust indices to account for chunk offset
        closest_indices[update_mask] = chunk_argmin[update_mask] + chunk_start

    return closest_indices

### ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###

@click.command()
@click.argument('testfunction', type=str, required=True)
def CLI(testfunction):
    
    scenario = 'baf_test_new_Iter0'
    year = 2050
    commodities = ["HEAT", "HYDROGEN"]

    if testfunction == "fitting_supply_curves":
        # Example usage for one scenario
        resulting_curves = {}
        
        ## Get Balmorel Results
        gams_system_directory = '/opt/gams/48.5'
        m=Balmorel('Balmorel')
        m.locate_results()
        res = MainResults('MainResults_' + scenario + '.gdx',
                        paths='Balmorel/' + m.scname_to_scfolder[scenario] + '/model', 
                        system_directory=gams_system_directory)
        production = res.get_result('F_CONS_YCRAST')
        el_prices= res.get_result('EL_PRICE_YCRST')
        idx = el_prices.query('Value < 1e-6').index
        el_prices.loc[idx, 'Value'] = 0 
        
        ## Prepare parameters to fit
        parameters = el_prices.round({'Value' : 0}).pivot_table(index=['Region', 'Season', 'Time'], values='Value').reset_index().rename(columns={'Value' : 'Elprice'})
        
        ## Make curves
        for commodity in commodities:  
            resulting_curves[commodity] = get_supply_curves(scenario, year, commodity, parameters, production, el_prices, -1, plot_overall_curves=True)
    
    elif testfunction == 'model_supply_curves': 

        test_data = {}        
        for parameter in ['all_parameters', 'supply_curves']:
            with open(f'../tests/{parameter}_for_model.pkl', 'rb') as f:
                test_data[parameter] = pickle.load(f)
            
        antares_input = AntaresInput()
        region = "DE"
        unserved_energy_cost = configparser.ConfigParser()
        unserved_energy_cost.read("Antares/input/thermal/areas.ini")
        weather_years = [
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
        # weather_years = [2000]
        commodity = "HEAT"

        unserved_energy_cost = model_supply_curves_in_antares(
            weather_years,
            test_data["all_parameters"],
            test_data["supply_curves"][commodity],
            antares_input,
            commodity,
            region,
            unserved_energy_cost,
        )


if __name__ == "__main__":
    CLI()
