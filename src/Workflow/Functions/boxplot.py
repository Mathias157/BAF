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


def plot_boxplot(df: pd.DataFrame):
    """
    Plot boxplot of values where TestYear != TrainYear, grouped by Region.
    
    Parameters:
    df: DataFrame with columns Value, Region, TestYear, TrainYear
    """
    # Filter for rows where TestYear != TrainYear
    filtered_df = df[df['TestYear'] != df['TrainYear']].copy()
    
    # Get unique regions for x-axis
    regions = sorted(pd.Series(filtered_df['Region']).unique().tolist())
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for boxplot - one box per region
    data_for_plot = []
    for region in regions:
        region_data = np.array(filtered_df[filtered_df['Region'] == region]['Value'].tolist())
        data_for_plot.append(region_data)
    
    # Create boxplot
    bp = ax.boxplot(data_for_plot, patch_artist=True)
    
    # Set x-axis labels
    ax.set_xticklabels(regions)
    
    # Style the boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Add labels and title
    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel('Relative Difference (%)', fontsize=12)
    ax.set_title('PtX Demand Deviations (Test Year ≠ Train Year)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if many regions
    if len(regions) > 10:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig, ax


# ------------------------------- #
#            2. Main              #
# ------------------------------- #


@click.command()
def main():
    df = collect_and_concat_dataframes()
    
    # Plot the boxplot for values where TestYear != TrainYear
    fig, ax = plot_boxplot(df)
    plt.show()


if __name__ == "__main__":
    main()
