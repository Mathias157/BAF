"""
Created on 30/03/2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)

Adapted March 2026

IN ONE SENTENCE:
Handles BALMOREL-BALMOREL coupling
"""

# ------------------------------- #
#        0. Script Settings       #
# ------------------------------- #

from pathlib import Path
from pybalmorel import MainResults
import pandas as pd
import numpy as np
import subprocess
import click
import os
import platform
import shutil

# ------------------------------- #
#          1. Functions           #
# ------------------------------- #

# A function that increases from -0.9 at x = 0 to 0 at x = 2.5
decrease_function = lambda x: 0.9/2.5*x - 0.9

# Fictive demand calculation, adapated from BAF/src/Workflow/Functions/Methods
def fictdem_existing_ts(BalmArea: str,
                        year: str,
                        fict_de_factor: str,
                        fDEVAR: pd.DataFrame,
                        ENS: pd.DataFrame, 
                        LOLE: float,
                        i: int, 
                        negative_feedback: bool = True):

    # Prepare table
    try: 
        fDEVAR.loc[(year, BalmArea), :]
    except KeyError:
        idx = pd.MultiIndex.from_product([[year], [BalmArea]])
        fDEVAR = pd.concat((fDEVAR, pd.DataFrame(index=idx,
                                                  columns=['Value'],
                                                  data=[0])))

    if LOLE > 3:
        total_ENS = ENS.loc[i].sum()
        fDEVAR.loc[(year, BalmArea), :] += total_ENS*eval(fict_de_factor) / len(fDEVAR.columns)
        print('%s adding elfictdem: '%BalmArea, total_ENS*eval(fict_de_factor))
    elif (LOLE < 2.5) and negative_feedback and (i != 0):
        last_fictdem = ENS.loc[i-1].sum()
        previous_factor = eval(fict_de_factor.replace('i', '(i-1)'))
        subtraction = last_fictdem *decrease_function(LOLE)*previous_factor / len(fDEVAR.columns)
        print('%s subtracting elfictdem: '%BalmArea, subtraction*len(fDEVAR.columns))
        fDEVAR.loc[(year, BalmArea), :] += float(subtraction) # decrease_function is negative, so we are adding a negative number
        
        # Make sure it's not negative (can happen if there was a small ENS but no fictdem added because LOLE is < 3 h)
        if np.all(fDEVAR.loc[(year, BalmArea), :] < 0):
            fDEVAR.loc[(year, BalmArea), :] = 0 
    else:
        print('Didnt add FICTDE for %s %s because EL LOLE %0.2f'%(year, BalmArea, LOLE))
        pass
    
    return fDEVAR

# ------------------------------- #
#            2. Main              #
# ------------------------------- #

@click.group()
@click.pass_context
def main(
    ctx, scenario: str = "test", scenario_folder: str = "base", iteration: int = 0
):
    """CLI for running bi-directional BALMOREL-BALMOREL"""

    # Figure out operating system
    OS = platform.platform().split("-")[0]  # Assuming that linux will be == HPC!

    if not Path("bifiles").exists():
        Path("bifiles").mkdir()

    # Iteration Meta and Overall Results
    ENS = Path(f"bifiles/ENS_{scenario}_{iteration}.csv")

    if not ENS.exists():
        ENS = pd.DataFrame(columns=["Year","R","Commodity","S","T","Value","iteration"])
        ENS.to_csv(f"bifiles/ENS_{scenario}_{iteration}.csv")

    # Store to context
    ctx.ensure_object(dict)
    ctx.obj["OS"] = OS
    ctx.obj["iteration"] = iteration
    ctx.obj["scenario"] = scenario
    ctx.obj["scenario_folder"] = scenario_folder


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

    ENS = pd.read_csv(f"bifiles/ENS_{scenario}_{iteration}.csv")
    for weather_year in [2000]:
        # Load MainResults
        res = MainResults(
            f"Balmorel/{ctx.obj['scenario_folder']}/model/MainResults_{ctx.obj['scenario']}_operational_Iter{iteration}.gdx"
        )

        # Calculate energy not supplied
        temp = (
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
        temp['weather_year'] = weather_year 
        temp['iteration'] = iteration
        ENS = pd.concat((ENS, temp))

    # Assess adequacy for all carriers
    carriers = ENS.Commodity.unique()
    ELOLE = []
    for year in ENS.Year.unique():
        for region in ENS.R.unique():
            for carrier in carriers:

                # Count hours of backup production
                LOLE = ENS.query(f"Year == '{year}' and R =='{region}' and Commodity == '{carrier}'").shape[0]
                print(f"{carrier} LOLE in {region}: \t", LOLE, "h")

                if carrier == "ELECTRICITY":
                    ELOLE.append(LOLE)

    convergence = np.all(np.array(ELOLE) <= 3)
    print("\nConvergence achieved: %s\n" % convergence)

    # Store results
    ENS.to_csv(f"bifiles/ENS_{scenario}_{iteration}.csv")
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
    ENS = pd.read_csv(f"bifiles/ENS_{scenario}_{iteration}.csv")

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
