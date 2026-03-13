"""
Created on 30/03/2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)

Adapted March 2026

IN ONE SENTENCE:
Handles BALMOREL-BALMOREL coupling
"""

from pathlib import Path
from pybalmorel import MainResults
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
                "Y",
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
        .groupby(by=["R", "Commodity", "S", "T"])
        .aggregate({"Value": np.sum})
        .reset_index()
    )

    # Assess adequacy for all carriers
    carriers = ENS.Commodity.unique()
    for BalmArea in ENS.R.unique():
        idx1 = ENS.R == BalmArea

        for carrier in carriers:
            idx2 = ENS.Commodity == carrier

            # Count hours of backup production
            LOLE = ENS[idx1 & idx2].shape[0]
            print(f"{carrier} LOLE in {BalmArea}: \t", LOLE, "h")

            if carrier == "ELECTRICITY":
                ELOLE.loc[ctx, BalmArea] = LOLE

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
    EENS = pd.read_csv("Workflow/OverallResults/%s_ElecNotServedMWh.csv" % scenario)
    H2ENS = pd.read_csv("Workflow/OverallResults/%s_H2NotServedMWh.csv" % scenario)
    ELOLE = pd.read_csv("Workflow/OverallResults/%s_ElecLOLE.csv" % scenario)

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
    inv_res = MainResults(
        f"Balmorel/{ctx.obj['scenario_folder']}/model/MainResults_{ctx.obj['scenario']}_investment_Iter{iteration}.gdx"
    )
    balm_t = inv_res.get_result(
        "EL_PRICE_YCRST", ["Y", "C", "R", "S", "T", "Unit", "Val"]
    ).groupby(by=["S", "T"])
    balm_t = balm_t.aggregate({"Val": np.sum}).reset_index()[["S", "T"]]
    S = [f"S{i:02.0f}" % i for i in range(1, 54)]
    T = [f"T{i:03.0f}" % i for i in range(1, 169)]
    idx = pd.MultiIndex.from_product([S, T], names=["S", "T"])[:8760]
    hour = pd.DataFrame(data=np.arange(8760), columns=["Hour"], index=idx)

    print("%d seasons:" % len(balm_t.S.unique()), balm_t.S.unique())
    print("%d terms:" % len(balm_t["T"].unique()), balm_t["T"].unique())

    for BalmArea in ENS.R.unique():
        idx1 = ENS.R == BalmArea

        for carrier in ENS.loc[idx1, "Commodity"].unique():
            idx2 = ENS.Commodity == carrier

            # Get series of LOLE (backup capacity production)
            LOLE = (
                ENS.loc[idx1 & idx2].groupby(by=["S", "T"]).aggregate({"Value": np.sum})
            )  # Not summing, just re-arranging

            if carrier == "ELECTRICITY":
                # Join to full hour
                t = hour.copy()
                t = t.join(LOLE)
                t = t.fillna(0).reset_index()

                front_t = t.copy().drop(columns=["S", "T"])
                front_t.index = np.arange(-len(t), 0)
                back_t = t.copy().drop(columns=["S", "T"])
                back_t.index = np.arange(len(t), 2 * len(t))
                temp_t = pd.concat((front_t, t.copy(), back_t))

                for n, row in balm_t.iterrows():
                    # Get current hour
                    h1 = temp_t.loc[
                        (temp_t["S"] == row["S"]) & (temp_t["T"] == row["T"]), "Hour"
                    ].values[0]

                    # Get previous hour
                    try:
                        h0 = temp_t.loc[
                            (temp_t["S"] == balm_t.loc[n - 1, "S"])
                            & (temp_t["T"] == balm_t.loc[n - 1, "T"]),
                            "Hour",
                        ].values[0]
                        dif0 = round((h1 - h0) / 2)
                    except KeyError:
                        h0 = temp_t.loc[
                            (temp_t["S"] == balm_t.loc[len(balm_t) - 1, "S"])
                            & (temp_t["T"] == balm_t.loc[len(balm_t) - 1, "T"]),
                            "Hour",
                        ].values[0]
                        dif0 = round((h1 + len(t) - h0) / 2)

                    # Get last hour
                    try:
                        h2 = temp_t.loc[
                            (temp_t["S"] == balm_t.loc[n + 1, "S"])
                            & (temp_t["T"] == balm_t.loc[n + 1, "T"]),
                            "Hour",
                        ].values[0]
                        dif2 = round((h2 - h1) / 2)
                    except KeyError:
                        h2 = temp_t.loc[
                            (temp_t["S"] == balm_t.loc[0, "S"])
                            & (temp_t["T"] == balm_t.loc[0, "T"]),
                            "Hour",
                        ].values[0]
                        dif2 = round((h2 + len(t) - h1) / 2)

                    # Accumulated Unsupplied Energy
                    idx = (temp_t.index >= h1 - dif0) & (temp_t.index < h1 + dif2 - 1)
                    balm_t.loc[n, BalmArea + "_UNSELEC"] = temp_t.loc[
                        idx, "Value"
                    ].sum()

                agg = balm_t.groupby(by=["S"])
                agg = agg.aggregate({BalmArea + "_UNSELEC": np.sum})

                if iteration != 0:
                    fDEVAR[BalmArea] += agg[BalmArea + "_UNSELEC"] * fictdemfactor
                else:
                    fDEVAR[BalmArea] = agg[BalmArea + "_UNSELEC"]

                # Store overall unserved energy
                EENS.loc[iteration, BalmArea] = agg[BalmArea + "_UNSELEC"].sum()

            else:
                if iteration != 0:
                    fDH2VAR[BalmArea] += (
                        t.groupby(by="S").aggregate({"Value": np.sum}).Value
                    )
                else:
                    fDH2VAR[BalmArea] = (
                        t.groupby(by="S").aggregate({"Value": np.sum}).Value
                    )

                # Store overall unserved energy
                H2ENS.loc[iteration, BalmArea] = (
                    LOLE.groupby(by="S").aggregate({"Value": np.sum}).sum().values[0]
                )

    ### 3.5 Create Balmorel Files
    FICTDE = ""
    FICTDH2 = ""
    FICTDE_VAR_T = ""
    for BalmArea in ENS.R.unique():
        FICTDE = FICTDE + "DE('2050','%s','FICTIVE') = %0.2f;\n" % (
            BalmArea,
            fDEVAR[BalmArea].sum(),
        )
        FICTDH2 = (
            FICTDH2
            + "HYDROGEN_DH2('2050','%s') = HYDROGEN_DH2('2050','%s') + %0.2f;\n"
            % (BalmArea, BalmArea, fDH2VAR[BalmArea].sum())
        )

        for season in fDEVAR[BalmArea].index:
            FICTDE_VAR_T = (
                FICTDE_VAR_T
                + "DE_VAR_T('%s', 'FICTIVE', '%s', TTT) = %d/168;\n"
                % (BalmArea, season, fDEVAR.loc[season, BalmArea])
            )

    if use_fictdem:
        with open("Balmorel/base/data/ANTBALM_FICTDE.inc", "w") as f:
            f.write(FICTDE)

        with open("Balmorel/base/data/ANTBALM_FICTDH2.inc", "w") as f:
            f.write(FICTDH2)

        with open("Balmorel/base/data/ANTBALM_FICTDE_VAR_T.inc", "w") as f:
            f.write(FICTDE_VAR_T)

        fDEVAR.to_csv("MetaResults/FICTDEprofile.csv")
        fDH2VAR.to_csv("MetaResults/FICTDH2profile.csv")

    ### 3.6 Save Results for Next Iteration
    EENS.to_csv("OverallResults/%s_ElecNotServedMWh.csv" % scenario)
    H2ENS.to_csv("OverallResults/%s_H2NotServedMWh.csv" % scenario)
    ELOLE.to_csv("OverallResults/%s_ElecLOLE.csv" % scenario)


if __name__ == "__main__":
    main()
