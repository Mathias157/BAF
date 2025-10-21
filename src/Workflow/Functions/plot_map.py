import click
import numpy as np
import geopandas as gpd
from pathlib import Path
from matplotlib.patches import FancyArrowPatch, ArrowStyle 
from pybalmorel import MainResults
import pandas as pd
import matplotlib.pyplot as plt
from GeneralHelperFunctions import AntaresOutput

special_coordinates = {
    'NO' : [8.76, 61.34],
    'SE' : [15.02, 58.96]
}

midpoints = {
    'NO-DE' : [6.73, 55.84],
}

def get_transmission(results: MainResults, commodity: str):
    if commodity.lower() == "electricity":
        df = results.get_result("X_CAP_YCR")
    elif commodity.lower() == "hydrogen":
        df = results.get_result("XH2_CAP_YCR")
    else:
        raise ValueError("Choose electricity or hydrogen!")

    return df


@click.group()
@click.pass_context
@click.argument("scenario", type=str, required=True)
@click.option("--year", type=int, required=False, default=2050, help="Model year")
@click.option(
    "--scenario-folder",
    type=str,
    required=False,
    default="base",
    help="The scenario folder containing the scenario result file",
)
@click.option("--antares-scenario", type=str, default="", help="Use a specific Antares output, don't try to search")
@click.option("--mc-year", type=str, default='mc-all', help="MC year to load results from")
@click.option("--dark", is_flag=True, default=False, help="Make plots dark")
def main(ctx, scenario, year, scenario_folder, antares_scenario, mc_year, dark):
    ctx.ensure_object(dict)

    if dark:
        ctx.obj["facecolor"] = "none"
        plt.style.use("dark_background")
    else:
        ctx.obj["facecolor"] = "white"

    ctx.obj["gams_system_directory"] = "/appl/gams/47.6.0/"

    # Find MainResults
    ctx.obj["MainResults"] = MainResults(
        [f"MainResults_{scenario}.gdx"],
        paths=f"Balmorel/{scenario_folder}/model",
        system_directory=ctx.obj["gams_system_directory"],
    )
    ctx.obj["scenario"] = scenario
    ctx.obj["year"] = year

    # Find Antares result
    if antares_scenario == '':
        antares_scenario = scenario

    antares = Path('Antares/output')
    antares_result = [result.name for result in antares.glob(f'*eco-{antares_scenario.lower()}_*')]
    if len(antares_result) > 1:
        raise ValueError("More than one Antares result!")
    else:
        antares_result = antares_result[0]

    ctx.obj['AntaresOutput'] = AntaresOutput(result_name=antares_result)
    ctx.obj['mc_year'] = mc_year


@main.command()
@click.pass_context
@click.argument("commodity", type=str, required=True)
def capacities(ctx, commodity):
    res = ctx.obj["MainResults"]
    scenario = ctx.obj["scenario"]
    year = ctx.obj["year"]

    df = get_transmission(res, commodity).query(f'Year == "{year}"')

    max_cap = df.Value.max()
    gf = gpd.read_file("2025AntBalmMap.geojson")

    # Plot
    fig, ax = plt.subplots(dpi=200, figsize=(15, 10))
    gf.plot(ax=ax)

    for i, row in df.iterrows():
        c0 = gf.query(f"""id == '{row["From"]}' """).centroid
        c1 = gf.query(f"""id == '{row["To"]}' """).centroid

        ax.plot([c0.x, c1.x], [c0.y, c1.y], linewidth=row["Value"], color="g")

    fig.savefig("test.png")

@click.pass_context
def get_balmorel_flows(ctx):
    res = ctx.obj["MainResults"]
    year = ctx.obj["year"]

    df_cap = get_transmission(res, "electricity").query(f'Year == "{year}"')
    unique_capacities = df_cap.pivot_table(index="From", columns="To", values="Value")
    df_flow = (
        res.get_result("X_FLOW_YCR")
        .query(f'Year == "{year}"')
        .pivot_table(index="From", columns="To", values="Value", aggfunc='sum')
    )

    return unique_capacities, df_flow

@click.pass_context
def get_antares_flows(ctx):
   
    year = ctx.obj["year"]
    res = ctx.obj['MainResults']
    ant_res = ctx.obj['AntaresOutput']
    df_cap = get_transmission(res, "electricity").query(f'Year == "{year}"')
    unique_capacities = df_cap.pivot_table(index="From", columns="To", values="Value")

    data = []
    for region_from in unique_capacities.index:
        for region_to in unique_capacities.columns:
            if unique_capacities.loc[region_from, region_to] > 0:
                try:
                    df = ant_res.load_link_results([region_from, region_to],
                                        temporal='annual',
                                        mc_year=ctx.obj['mc_year'])
                    net_flow = df.loc[:, 'FLOW LIN.'].sum()/1e6
                    if net_flow > 0:
                        data.append([region_from, region_to, net_flow])
                    else:
                        data.append([region_to, region_from, -net_flow])
                except FileNotFoundError:
                    pass

    df_flow_from = pd.DataFrame(
        data,
        columns=['From', 'To', 'Value']
    )
    df_flow_to = pd.DataFrame(
        data,
        columns=['To', 'From', 'Value']
    )
    df_flow_to.Value = 0
    df_flow = (
        pd.concat(
                (df_flow_from,
                df_flow_to),
            )
        .pivot_table(
        index="From",
        columns="To",
        values="Value",
        aggfunc='sum'
        )
    )

    return unique_capacities, df_flow

@main.command()
@click.pass_context
@click.argument('model', type=str, default='balmorel')
def flow(ctx, model: str = 'Balmorel'):
    if model.lower() == 'balmorel':
        unique_capacities, df_flow = get_balmorel_flows()
    else:
        unique_capacities, df_flow = get_antares_flows()

    filename = f"{ctx.obj['scenario']}_{model}_elflows.png"
    plot_elflow(unique_capacities, df_flow, filename)

def plot_elflow(unique_capacities, df_flow, filename):

    # Plot
    if 'dispatch_WY2000' in filename:
        gf = gpd.read_file("Pre-Processing/2025AntBalmMap.geojson").query('id in ["FR", "DE", "ES"]')
    else:
        gf = gpd.read_file("Pre-Processing/2025AntBalmMap.geojson").query('ADMIN != "Ukraine" and ADMIN != "Turkey" and ADMIN != "Belarus"')

    fig, ax = plt.subplots(dpi=200, figsize=(15, 10))
    gf.plot(ax=ax, facecolor='grey')
    scaling = 5
    all_net_flows = []

    for region_from in unique_capacities.index:
        for region_to in unique_capacities.columns:
            if (
                unique_capacities.loc[region_from, region_to] > 0
            ):
                # Calculate net flow
                try:
                    net_flow = (
                        df_flow.loc[region_from, region_to]
                        - df_flow.loc[region_to, region_from]
                    )
                except KeyError:
                    try:
                        net_flow = df_flow.loc[region_from, region_to]
                    except KeyError:
                        print(f'No flow from {region_from} to {region_to} despite non-zero capacity')
                        continue

                # Skip, if net flow is in the other direction
                if net_flow < 0:
                    continue

                # Find coordinates
                if region_from not in special_coordinates.keys():
                    c1 = gf.query(f"id == '{region_from}' ").centroid.values[0]
                    x1, y1 = c1.x, c1.y
                else:
                    x1, y1 = special_coordinates[region_from]

                if region_to not in special_coordinates.keys():
                    c2 = gf.query(f"id == '{region_to}' ").centroid.values[0]
                    x2, y2 = c2.x, c2.y
                else:
                    x2, y2 = special_coordinates[region_to]

                if (f'{region_from}-{region_to}' in midpoints.keys() or 
                    f'{region_to}-{region_from}' in midpoints.keys()
                        ):
                    try:
                        c = midpoints[f'{region_from}-{region_to}']
                    except KeyError:
                        c = midpoints[f'{region_to}-{region_from}']
                    x_coords = [x1, c[0], x2]
                    y_coords = [y1, c[1], y2]
                else:
                    x_coords = [x1, x2]
                    y_coords = [y1, y2]

                print(f"Flow from {region_from} to {region_to}: {net_flow} TWh")

                # Make arrow if flow is larger than 10 TWh
                if net_flow > 10:
                    style = ArrowStyle("Fancy", head_length=4, head_width=8, tail_width=0.1)
                    arrow = FancyArrowPatch(
                        (x1 + 0.49 * (x_coords[1] - x1), y1 + 0.49 * (y_coords[1] - y1)),
                        (x1 + 0.501 * (x_coords[1] - x1), y1 + 0.501 * (y_coords[1] - y1)),
                        arrowstyle=style,
                        color="black",
                        zorder=5,
                    )
                    ax.add_patch(arrow)

                # Draw arrow
                ax.plot(
                    x_coords, y_coords, 
                    linewidth=abs(net_flow) / scaling, 
                    color="lightblue"
                )

                # Store net flow
                all_net_flows.append(net_flow)

    # Set title
    max_flow = np.max(all_net_flows)
    min_flow = np.min(all_net_flows)
    ax.set_title(f'Max flow: {max_flow:0.2f} TWh, min flow: {min_flow:0.2f} TWh')
    fig.savefig(f'Workflow/OverallResults/{filename}')


@main.command()
@click.pass_context
@click.argument("commodity", type=str, required=True)
def pybalm(ctx, commodity):
    res = ctx.obj["MainResults"]
    scenario = ctx.obj["scenario"]
    year = ctx.obj["year"]

    fig, _ = res.plot_map(
        scenario=scenario,
        year=year,
        commodity=commodity,
        lines="FlowYear",
        generation="Production",
        path_to_geofile="./Pre-Processing/2025AntBalmMap.geojson",
        background="H2 Net Export",
        pie_value_max=1000000,
    )

    fig.savefig(
        f"Workflow/OverallResults/map_{scenario}_{commodity}_{year}.png",
        transparent=True,
    )


if __name__ == "__main__":
    main()
