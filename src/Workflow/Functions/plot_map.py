import click
import geopandas as gpd
from pybalmorel import Balmorel, MainResults
import matplotlib.pyplot as plt

def get_transmission(results: MainResults, commodity: str):
    if commodity.lower() == 'electricity':
        df = results.get_result('X_CAP_YCR')
    elif commodity.lower() == 'hydrogen':
        df = results.get_result('XH2_CAP_YCR')
    else:
        raise ValueError("Choose electricity or hydrogen!")

    return df

@click.group()
@click.pass_context
@click.option('--dark', is_flag=True, default=False, help='Make plots dark')
def main(ctx, dark):
    
    ctx.ensure_object(dict)
    
    if dark:
        ctx.obj['facecolor']='none'
        plt.style.use('dark_background')
    else:
        ctx.obj['facecolor']='white'
    

@click.command()
@click.argument('scenario', type=str, required=True)
@click.argument('commodity', type=str, required=True)
@click.option('--year', type=int, required=False, default=2050, help="Model year")
@click.option('--scenario-folder', type=str, required=False, default='base', help="The scenario folder containing the scenario result file")
def homemade(scenario, commodity, year, scenario_folder):

   mr = MainResults([f'MainResults_{scenario}.gdx'],
                    paths=f'Balmorel/{scenario_folder}/model')

   df = get_transmission(mr, commodity).query(f'Year == "{year}"')

   max_cap = df.Value.max()

   gf = gpd.read_file('2025AntBalmMap.geojson')

   fig, ax = plt.subplots(dpi=200, figsize=(15,10))
   gf.plot(ax=ax)

   for i, row in df.iterrows():
       c0 = gf.query(f'''id == '{row["From"]}' ''').centroid
       c1 = gf.query(f'''id == '{row["To"]}' ''').centroid

       ax.plot([c0.x, c1.x],
               [c0.y, c1.y],
               linewidth=row['Value'], color='g')

   fig.savefig('test.png')

@main.command()
@click.argument('scenario', type=str)
@click.argument('commodity', type=str, required=True)
@click.option('--year', type=int, required=False, default=2050, help="Model year")
@click.option('--scenario-folder', type=str, required=False, default='base', help="The scenario folder containing the scenario result file")
def pybalm(scenario, commodity, year, scenario_folder):

    res = MainResults([f'MainResults_{scenario}.gdx'],
                        paths=f'Balmorel/{scenario_folder}/model')
    fig, _ = res.plot_map(scenario=scenario, year=year,
                          commodity=commodity, lines='FlowYear', 
                          generation='Production', path_to_geofile='./2025AntBalmMap.geojson',
                          background='H2 Net Export',
                            pie_value_max = 1000000
                          )

    fig.savefig(f'Workflow/OverallResults/map_{scenario}_{commodity}_{year}.png', transparent=True)

if __name__ == '__main__':
    main()
