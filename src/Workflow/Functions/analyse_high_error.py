"""
An analysis of the regions with high error

Histograms will reveal which parameters show 
the highest errors, based on some tolerance.
Then, more deep-diving analyses will see if there 
are any correlations between the results of 
runs that work and runs that don't.

Created on 26.09.2025
@author: Mathias Berg Rosendal
         PhD Student at DTU Management (Energy Economics & Modelling)
"""
# ------------------------------- #
#        0. Script Settings       #
# ------------------------------- #

import matplotlib.pyplot as plt
import click
import pandas as pd
from boxplot import collect_and_concat_dataframes
from Formatting import cmcrameri_style
from pybalmorel import MainResults
from pathlib import Path
from pybalmorel.formatting import balmorel_colours

# ------------------------------- #
#          1. Functions           #
# ------------------------------- #

def get_high_error_index(tol: float = 20.0):

    df = collect_and_concat_dataframes()
    df = df.query(f"Value.abs() >= {tol}")

    return df

def get_mainresults_paths(balmorel_folder: str = 'Balmorel'):
    paths = (
        Path(balmorel_folder)
        .glob('./**/model/MainResults_*.gdx')
    )
    return paths

def plot_barchart(df: pd.DataFrame, 
                  index: list | str,
                  plot_name: str, 
                  **plot_kwargs):
    fig, ax = plt.subplots(figsize=plot_kwargs.get('figsize'))
    (
        df
        .query('not Generation.str.contains("BACKUP")')
        .pivot_table(index=index,
                    columns='Technology',
                    values='Value',
                    aggfunc='sum')
        .plot(
            ax=ax,
            kind='bar',
            stacked=True,
            color=balmorel_colours,
            **plot_kwargs
        )
    )
    ax.legend(loc='center left', bbox_to_anchor=(1.01, .5))
    fig.savefig(plot_name, bbox_inches='tight')


# ------------------------------- #
#            2. Main              #
# ------------------------------- #


@click.group()
@click.pass_context
@click.option("--tol", type=float, default=20.0, help="The tolerance for selecting 'high error'")
@click.option("--dark", is_flag=True, default=False, help="Make plots dark")
@click.option("--allwy", is_flag=True, default=False, help="Plot all weather years?")
def main(ctx, tol, dark, allwy):
    ctx.ensure_object(dict)

    if dark:
        ctx.obj["facecolor"] = "none"
        plt.style.use("dark_background")
    else:
        ctx.obj["facecolor"] = "white"
    cmcrameri_style(dark=dark)

    ctx.obj['tolerance'] = tol

    command = ctx.invoked_subcommand
    if command in ['barcharts']:

        if not(allwy):
            ctx.obj['results'] = MainResults(
                [file.name for file in get_mainresults_paths() if 'WY2000' in file.name],
                [str(file.parent) for file in get_mainresults_paths() if 'WY2000' in file.name]
            )
            ctx.obj['allwy'] = False
        else:
            ctx.obj['results'] = MainResults(
                [file.name for file in get_mainresults_paths()],
                [str(file.parent) for file in get_mainresults_paths()] 
            )
            ctx.obj['allwy'] = True



@main.command()
@click.pass_context
def histograms(ctx):

    df = get_high_error_index(ctx.obj['tolerance'])

    for category in ["Scenario", "Category", "Region", "TestYear", "TrainYear"]:
        df[category].hist()
        plt.show()


@main.command()
@click.pass_context
@click.option('--region', type=str, default='all', help="Regional scope, defaults to 'all'")
def barcharts(ctx, region):
    
    for result in ['G_STO_YCRAF', 'G_CAP_YCRAF', 'PRO_YCRAGF']:
        df = ctx.obj['results'].get_result(result)
        
        # Filters
        if result == 'G_CAP_YCRAF':
            df = df.query('not Technology.str.contains("STORAGE")')
        if region.lower() != 'all':
            df = df.query(f'Region == "{region}"')

        if not(ctx.obj['allwy']):
            plot_barchart(df, 'Scenario', 
                        f'Workflow/OverallResults/{result}_{region}_barchart.png')
        elif result == 'PRO_YCRAGF':
            for scenario in ['noh', 'noh2', 'h2', 'h2_lss', 'h2_lss_h2t']:
                plot_barchart(df.query(f'Scenario.str.contains("{scenario}_dispatch") and Commodity == "ELECTRICITY"'),
                              'Scenario',
                              f'Workflow/OverallResults/{scenario}_{result}_{region}_barchart.png',
                              figsize=(20,5))


if __name__ == "__main__":
    main()
