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
from matplotlib.patches import Patch
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
    absolute_difference: bool = True,
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

    if absolute_difference:
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
            absolute_difference=False
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

def plot_boxplot(df: pd.DataFrame, title: str, average_regions: bool = False):
    """
    Plot boxplot of values with separate boxes for test years and train years
    
    Parameters:
    df: DataFrame with columns Value, Region, TestYear, TrainYear
    """
    # Mark train and test years
    df.loc[:, 'YearType'] = 'Test Years'
    df.loc[df.eval('TestYear == TrainYear'), 'YearType'] = 'Train Years'
    
    if average_regions:
        # Do average across regions 
        df = df.pivot_table(
            index=list(df.columns.drop(['Region', 'Value'])),
            values='Value',
            aggfunc=lambda x: np.mean(np.abs(x))
        ).reset_index()
    
    # Get unique scenarios for x-axis
    scenarios = ['NoH', 'NoH2', 'H2', 'H2LSS', 'H2LSSH2T']
    df = df.replace({
        'noh' : 'NoH',
        'noh2' : 'NoH2',
        'h2' : 'H2',
        'h2_lss' : 'H2LSS',
        'h2_lss_h2t' : 'H2LSSH2T',
    })
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Prepare data for boxplot - separate data for test years and train years
    test_data = []
    train_data = []
    
    for scenario in scenarios:
        # Test years data
        test_scenario_data = df.query('Scenario == @scenario and YearType == "Test Years"').Value.tolist()
        test_data.append(test_scenario_data)
        
        # Train years data  
        train_scenario_data = df.query('Scenario == @scenario and YearType == "Train Years"').Value.tolist()
        train_data.append(train_scenario_data)
    
    # Calculate positions for the boxplots (side by side)
    positions_test = [i - 0.2 for i in range(1, len(scenarios) + 1)]
    positions_train = [i + 0.2 for i in range(1, len(scenarios) + 1)]
    
    # Create boxplots
    bp1 = ax.boxplot(test_data, positions=positions_test, widths=0.35, 
                     patch_artist=True) 
    bp2 = ax.boxplot(train_data, positions=positions_train, widths=0.35, 
                     patch_artist=True)
    
    # Style the boxplots with different colors
    for patch in bp1['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    for patch in bp2['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Set x-axis labels at the center positions
    ax.set_xticks(range(1, len(scenarios) + 1))
    ax.set_xticklabels(scenarios)
    
    # Create legend
    legend_elements = [Patch(facecolor='lightcoral', alpha=0.7, label='Test Years'),
                      Patch(facecolor='lightblue', alpha=0.7, label='Train Years')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add labels and title
    ax.set_title(title, fontsize=14)
    if not(average_regions):
        ax.set_ylabel('Relative Difference (%)', fontsize=12)
        ax.set_ylim(-100, 100)
    else:
        ax.set_ylabel('Absolute Relative Difference (%)', fontsize=12)
        ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

    return df, fig, ax

# ------------------------------- #
#            2. Main              #
# ------------------------------- #


@click.command()
def main():
    df = collect_and_concat_dataframes()
    
    # Plot the boxplot for all data
    df, fig, ax = plot_boxplot(df, 'Endogenous electricity demands - average error for all regions and WY')
    fig.savefig('Workflow/OverallResults/boxplot_endodemand_comparison.png')

    # Plot the boxplot for system (aggregated absolute error across regions)
    df, fig, ax = plot_boxplot(df, 'Endogenous electricity demands - average absolute error for system for all WY', True)
    fig.savefig('Workflow/OverallResults/boxplot_endodemand_comparison_averageregions.png')

    # Print mean
    print('Mean of difference through all test years:    \n', df.query('YearType=="Test Years"').pivot_table(index='Scenario', columns='Category', values='Value', aggfunc='mean').round(2))
    print('Mean of difference through train years only:  \n', df.query('YearType=="Train Years"').pivot_table(index='Scenario', columns='Category', values='Value', aggfunc='mean').round(2))


if __name__ == "__main__":
    main()
