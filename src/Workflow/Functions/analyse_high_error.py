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
from boxplot import collect_and_concat_dataframes
from Formatting import cmcrameri_style
from pybalmorel import MainResults
from pathlib import Path

# ------------------------------- #
#          1. Functions           #
# ------------------------------- #

def get_high_error_index(tol: float = 20.0):

    df = collect_and_concat_dataframes()
    df = df.query(f"Value.abs() >= {tol}")

    return df

def get_mainresults_paths(balmorel_folder: str = 'Balmorel'):
    paths = (
        Path('Balmorel')
        .glob('./**/model/MainResults_*.gdx')
    )
    return paths

# ------------------------------- #
#            2. Main              #
# ------------------------------- #


@click.group()
@click.pass_context
@click.option("--tol", type=float, default=20.0, help="The tolerance for selecting 'high error'")
@click.option("--dark", is_flag=True, default=False, help="Make plots dark")
def main(ctx, tol, dark):
    ctx.ensure_object(dict)

    if dark:
        ctx.obj["facecolor"] = "none"
        plt.style.use("dark_background")
    else:
        ctx.obj["facecolor"] = "white"
    cmcrameri_style(dark=dark)

    ctx.obj['tolerance'] = tol

    command = ctx.invoked_subcommand
    if command in ['storage']:
        ctx.obj['results'] = MainResults(
            [file.name for file in get_mainresults_paths()],
            [str(file.parent) for file in get_mainresults_paths()]
        )


@main.command()
@click.pass_context
def histograms(ctx, tol):

    df = get_high_error_index(tol)

    for category in ["Scenario", "Category", "Region", "TestYear", "TrainYear"]:
        df[category].hist()
        plt.show()


@main.command()
@click.pass_context
def storage(ctx):
    
    df = ctx.obj['results'].get_result('G_CAP_YCRAF')
    
    fig, ax = plt.subplots()
    (
        df
        .pivot_table(index='Scenario',
                     columns='Technology',
                     values='Value',
                     aggfunc='sum')
        .plot(
            ax=ax,
            kind='bar',
            stacked=True
        )
    )
    plt.show()


if __name__ == "__main__":
    main()
