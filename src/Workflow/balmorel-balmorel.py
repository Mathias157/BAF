"""
Created on 30/03/2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)

Adapted March 2026

IN ONE SENTENCE:
Handles BALMOREL-BALMOREL coupling
"""

from pathlib import Path
from pybalmorel import MainResults
from Functions.Methods import fictdem_existing_ts
import pandas as pd
import numpy as np
import subprocess
import click
import os
import platform
import shutil


@click.group()
@click.pass_context
def main(
    ctx, scenario: str = "test", scenario_folder: str = "base", iteration: int = 0
):
    """CLI for running bi-directional BALMOREL-BALMOREL"""

    # Figure out operating system
    OS = platform.platform().split("-")[0]  # Assuming that linux will be == HPC!

    # Iteration Meta and Overall Results
    EENS, H2ENS, ELOLE = (
        Path(f"Workflow/OverallResults/{scenario}_ElecNotServedMWh.csv"),
        Path(f"Workflow/OverallResults/{scenario}_H2NotServedMWh.csv"),
        Path(f"Workflow/OverallResults/{scenario}_ElecLOLE.csv"),
    )

    if not Path("Workflow/OverallResults").exists():
        Path("Workflow/OverallResults").mkdir()

    if not EENS.exists() or not H2ENS.exists() or not ELOLE.exists():
        # Electricity not served
        fENS = pd.DataFrame({}, columns=["Iter", "Year", "Region", "Value (MWh)"])
        fENS.to_csv(EENS)

        # Hydrogen not served
        fENSH2 = pd.DataFrame({}, columns=["Iter", "Year", "Region", "Value (MWh)"])
        fENSH2.to_csv(H2ENS)

        # Loss of load durations
        fLOLD = pd.DataFrame(
            {}, columns=["Iter", "Year", "Region", "Carrier", "Value (h)"]
        )
        fLOLD.to_csv(ELOLE)
    else:
        fENS = pd.read_csv(EENS, index_col=0)
        fENS.loc[iteration, :] = np.zeros(fENS.shape[1])
        fENSH2 = pd.read_csv(H2ENS, index_col=0)
        fENSH2.loc[iteration, :] = np.zeros(fENSH2.shape[1])
        fLOLD = pd.read_csv(ELOLE, index_col=0)
        fLOLD.loc[iteration, :] = np.zeros(fLOLD.shape[1])

    # Store to context
    ctx.ensure_object(dict)
    ctx.obj["OS"] = OS
    ctx.obj["iteration"] = iteration
    ctx.obj["scenario"] = scenario
    ctx.obj["scenario_folder"] = scenario_folder
    ctx.obj["EENS"] = fENS
    ctx.obj["H2ENS"] = fENSH2
    ctx.obj["ELOLE"] = fLOLD


@main.command()
@click.pass_context
def peri_process(ctx):
    """Go from investment to operational run in scenario folder (should be three-step)"""

    print("\n----------------PERI-PROCESSING---------------\n")

    # Change balopt to operation
    balm_path = f"Balmorel/{ctx.obj['scenario_folder']}/model/"
    shutil.copyfile(balm_path + "balopt_operation.opt", balm_path + "balopt.opt")

    # Make full year time-series
    shutil.copyfile("Balmorel/base/data/T_operation.inc", "Balmorel/base/data/T.inc")
    shutil.copyfile("Balmorel/base/data/S_operation.inc", "Balmorel/base/data/S.inc")

    print("\nPeri-processing done\n----------------------------------------------\n")


@main.command()
@click.pass_context
def operation(ctx):
    """Operational BALMOREL run"""

    SC = f"{ctx.obj['scenario']}_operational_Iter{ctx.obj['iteration']}"

    # Running Balmorel Operation
    os.chdir(f"Balmorel/{ctx.obj['scenario_folder']}/model")
    if ctx.obj["OS"] == "Linux":
        Balm_cmd = ["gams", f'"Balmorel.gms" --scenario_name={SC}']
    else:
        Balm_cmd = f'gams "Balmorel.gms" --scenario_name={SC}'

    # Run it
    subprocess.run(Balm_cmd)


@main.command()
@click.pass_context
def convergence(ctx):
    """Check if capacities were adequate"""

    print("\n-------------CONVERGENCE-CRITERION------------\n")

    # Get context
    iteration = ctx.obj["iteration"]
    scenario = ctx.obj["scenario"]
    ELOLE = ctx.obj["ELOLE"]

    # Load MainResults
    res = MainResults(
        f"Balmorel/{ctx.obj['scenario_folder']}/model/MainResults_{ctx.obj['scenario']}_operational_Iter{iteration}.gdx"
    )

    # Calculate energy not supplied
    ENS = (
        res.get_result(
            "PRO_YCRAGFST",
            [
                "Year",
                "C",
                "R",
                "A",
                "G",
                "F",
                "S",
                "T",
                "Commodity",
                "Tech",
                "Unit",
                "Value",
            ],
        )
        .query('G.str.contains("BACKUP")')  # Only look at backup generation
        .groupby(by=["Year", "R", "Commodity", "S", "T"])
        .aggregate({"Value": np.sum})
        .reset_index()
    )
    ENS['iteration'] = iteration

    # Assess adequacy for all carriers
    carriers = ENS.Commodity.unique()
    for year in ENS.Year.unique():
        for region in ENS.R.unique():
            for carrier in carriers:

                # Count hours of backup production
                LOLE = ENS.query(f"Year == '{year}' and R =='{region}' and Commodity == '{carrier}'").shape[0]
                print(f"{carrier} LOLE in {region}: \t", LOLE, "h")

                if carrier == "ELECTRICITY":
                    ELOLE.loc[ctx, region] = LOLE

    convergence = np.all(ELOLE.loc[iteration] <= 3)
    print("\nConvergence achieved: %s\n" % convergence)

    # Store results
    ENS.to_csv(f"Workflow/OverallResults/ENS_{scenario}_{iteration}.csv")
    print(
        "Assessing convergence criterion done\n----------------------------------------------\n"
    )


@main.command()
@click.pass_context
@click.argument("strategy", type=str, default="fictdem")
@click.option(
    "--fictdemfactor",
    type=str,
    default=100,
    help="Factor on energy not served for fictive demand method",
)
def post_process(ctx, strategy: str, fictdemfactor: float = 100):
    """Post process operational results with incentive strategy"""

    # Get context
    iteration = ctx.obj["iteration"]
    scenario = ctx.obj["scenario"]
    ENS = pd.read_csv(f"Workflow/OverallResults/ENS_{scenario}_{iteration}.csv")

    print("\n### ---------------POST-PROCESSING---------------- ###\n")

    # Base factor on fictive electricity demand
    use_fictdem = True if "fictdem" in strategy.lower() else False
    use_capcred = True if "capcred" in strategy.lower() else False
    if iteration != 0 and use_fictdem:
        fDEVAR = pd.read_csv("MetaResults/FICTDEprofile.csv", index_col="S")
        fDH2VAR = pd.read_csv("MetaResults/FICTDH2profile.csv", index_col="S")
    else:
        fDEVAR = pd.DataFrame([])
        fDH2VAR = pd.DataFrame([])

    print("Loading timeseries from invest run...")

    # Load Balmorel timeseries index (electricity price is most light-weight) - this should be the low resolution timeseries though!
    for year in ENS.Year.unique():
        for region in ENS.R.unique():
            for carrier in ENS.Commodity.unique():

                if use_fictdem:
                    # Get series of LOLE (backup capacity production)
                    region_ENS = (
                        ENS
                        .query(f"Year == {year} and R == '{region}' and Commodity == '{carrier}'")
                        .pivot_table(index=["iteration", "S", "T"],
                                    values='Value',
                                    aggfunc='sum')
                    )  

                    # Skip if adequate
                    if len(region_ENS) == 0:
                        continue

                    LOLE = len(region_ENS.loc[iteration].index)

                    if carrier == "ELECTRICITY":

                        fDEVAR = fictdem_existing_ts(region, year, fictdemfactor, fDEVAR, region_ENS, LOLE, iteration)

                    else:
                        fDH2VAR = fictdem_existing_ts(region, year, fictdemfactor, fDH2VAR, region_ENS, LOLE, iteration)


    ### 3.5 Create Balmorel Files
    FICTDE = ""
    FICTDH2 = ""
    for year, region in fDEVAR.index:
        FICTDE = FICTDE + f"DE('{year}','{region}','FICTIVE') = {fDEVAR.loc[(year, region)].sum()};\n"  
            
    for year, region in fDH2VAR.index:
        FICTDH2 = (
            FICTDH2
            + "HYDROGEN_DH2('%d','%s') = HYDROGEN_DH2('2050','%s') + %0.2f;\n"
            % (year, region, region, fDH2VAR.loc[(year, region)].sum())
        )

    with open("Balmorel/base/data/ANTBALM_FICTDE.inc", "w") as f:
        f.write(FICTDE)

    with open("Balmorel/base/data/ANTBALM_FICTDH2.inc", "w") as f:
        f.write(FICTDH2)

    fDEVAR.to_csv("Workflow/MetaResults/FICTDEprofile.csv")
    fDH2VAR.to_csv("Workflow/MetaResults/FICTDH2profile.csv")


if __name__ == "__main__":
    main()
