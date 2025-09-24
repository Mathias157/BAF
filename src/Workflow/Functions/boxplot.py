"""
A -plot of the deviation in PtX demands

Plots the differences in endogenous electricity
demands between Antares and Balmorel fullyear runs
in boxes for all scenarios. One box using the deviation
based on results for all OTHER weather years than the one
that was used to model demand response in Antares, and one
only based on results from the weather year that WAS used
to model demand response in Antares.

Created on 24.09.2025
@author: Mathias Berg Rosendal
         PhD Student at DTU Management (Energy Economics & Modelling)
"""
# ------------------------------- #
#        0. Script Settings       #
# ------------------------------- #

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import click

# ------------------------------- #
#          1. Functions           #
# ------------------------------- #


def get_difference_table(
    filename: str,
    column_name: str,
    column_regex: str,
    column_elements_type: type,
    scenario_regex: str = "eco-(.+)_fullyear",
    column_to_extract_columns_from: str = "AntaresFile",
    column_to_extract_scenario_from: str = "AntaresFile",
    abslute_difference: bool = True,
):
    f = pd.read_csv(filename)

    # Get clustering amounts and scenario names
    f[column_name] = (
        f[column_to_extract_columns_from]
        .str.extract(column_regex)
        .astype(column_elements_type)
    )
    f["Scenario"] = f[column_to_extract_scenario_from].str.extract(scenario_regex)

    # Make table
    df_antares = f.query('Model=="Antares"').pivot_table(
        index=["Scenario", "Category"],
        columns=[column_name, "Region"],
        values="Value",
        aggfunc="sum",
    )
    df_balmorel = f.query('Model=="Balmorel"').pivot_table(
        index=["Scenario", "Category"],
        columns=[column_name, "Region"],
        values="Value",
        aggfunc="sum",
    )

    if abslute_difference:
        df_diff = ((df_antares - df_balmorel) / df_balmorel * 100).abs()
    else:
        df_diff = (df_antares - df_balmorel) / df_balmorel * 100

    # Aggregate to mean difference per region
    df_diff_mean = df_diff.groupby(level=0, axis=1).mean()

    return df_diff, df_diff_mean


def collect_and_concat_dataframes():

    collected_df = pd.DataFrame()
    for weather_year in [1982 + i for i in range(35)]:
        filename = f"Workflow/OverallResults/PtX_demand_comparison_multiweather_{weather_year}trained.csv"
        df, _ = get_difference_table(
            filename,
            "Data",
            r"\_dispatch\_WY(.+)\_Iter0",
            int,
            r"eco-(.+)\_wy",
            "BalmorelFile",
        )

        formatted_df = (
            df
            .stack()
            .stack()
            .reset_index(name="Value")
            .rename(columns={'Data' : 'TestYear'})
        ) 

        formatted_df['TrainYear'] = weather_year
        collected_df = pd.concat((collected_df, 
                                  formatted_df),
                                  ignore_index=True)

    return collected_df



# ------------------------------- #
#            2. Main              #
# ------------------------------- #


@click.command()
def main():
    df = collect_and_concat_dataframes()
    


if __name__ == "__main__":
    main()
