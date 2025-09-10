"""
Various Checks

A script for checking various stuff, see functions

Created on 08.09.2025
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
from GeneralHelperFunctions import AntaresOutput, ST_index, AntaresInput, log_time
from pybalmorel import MainResults

# ------------------------------- #
#          1. Functions           #
# ------------------------------- #

def demand_potentials(balmorel_result: MainResults,
                      antares_result: AntaresOutput,
                      commodity: str):

    if commodity.lower() == 'heat':
        category = 'ENDOGENOUS_ELECT2HEAT'
    else:
        category = 'ENDO_H2'

    df = (
        balmorel_result
        .get_result('EL_DEMAND_YCRST')
        .query(f'Category == "{category}"')
    ) 

    for region in df.Region.unique():

        temp = (
            df.query(f'Region == "{region}"')
            .pivot_table(index=['Season', 'Time'], values='Value', fill_value=0)
            .reindex(ST_index, fill_value=0)
        )

        # the total potential
        temp_ant = antares_result.load_area_results(f'{region}_{commodity}')['LOAD']

        # The realised demand
        temp_real = antares_result.load_link_results([region, f'{region}_{commodity}'])['FLOW LIN.']

        temp['Potential'] = temp_ant.values
        temp['Realised'] = temp_real.values

        print(region, commodity)
        print(temp)
        print('Net average difference:')
        print(f'Balmorel - Antares Pot.  = {(temp['Value']-temp['Potential']).sum()/8736:.02f} MW')
        print(f'Balmorel - Antares Real. = {(temp['Value']-temp['Realised']).sum()/8736:.02f} MW')

def prices(balmorel_result: MainResults,
           antares_result: AntaresOutput,
           commodity: str):

    df = (
        balmorel_result
        .get_result('EL_PRICE_YCRST')
    )

    for region in df.Region.unique():

        temp_balm = (
            df.query(f'Region == "{region}"')
            .pivot_table(index=['Season', 'Time'], values='Value')
            .reindex(ST_index, fill_value=0)
        )
        
        phys_ant = antares_result.load_area_results(f'{region}')['MRG. PRICE']
        virt_ant = antares_result.load_area_results(f'{region}_{commodity}')['MRG. PRICE']

        temp_balm['Phys.'] = phys_ant.values
        temp_balm['Virt.'] = virt_ant.values

        print(region, commodity)
        print(temp_balm)
        print('Net average difference:')
        print(f'Balmorel - Virtual  = {(temp_balm['Value']-temp_balm['Virt.']).sum()/8736:.02f} €/MWh')
        print(f'Balmorel - Physical = {(temp_balm['Value']-temp_balm['Phys.']).sum()/8736:.02f} €/MWh')

def compare_demands(balmorel_result,
                    antares_result1,
                    antares_result2):

    category = 'EXOGENOUS'

    df = (
        balmorel_result
        .get_result('EL_DEMAND_YCRST')
        .query(f'Category == "{category}"')
    ) 

    for region in df.Region.unique():

        temp = (
            df.query(f'Region == "{region}"')
            .pivot_table(index=['Season', 'Time'], values='Value', fill_value=0)
            .reindex(ST_index, fill_value=0)
        )

        # Exogenous load
        temp_ant1 = antares_result1.load_area_results(f'{region}')['LOAD']
        temp_ant2 = antares_result2.load_area_results(f'{region}')['LOAD']

        temp['Ant1'] = temp_ant1.values
        temp['Ant2'] = temp_ant2.values

        print(region)
        print(temp)
        # print('Net average difference:')
        # print(f'Balmorel - Antares Pot.  = {(temp['Value']-temp['Potential']).sum()/8736:.02f} MW')
        # print(f'Balmorel - Antares Real. = {(temp['Value']-temp['Realised']).sum()/8736:.02f} MW')

def get_availibility_input(region: str, commodity: str, hour: int):

    antares_input = AntaresInput()
    time_string = log_time().replace('[', '').replace(']:', '').replace(' ', '-').replace(':', '')

    weather_year=18
    availabilities = []
    for thermal_cluster in antares_input.thermal_clusters[region+'_'+commodity.lower()]:
        price = int(thermal_cluster.split('_')[0])
        demand = antares_input.thermal(region+'_'+commodity, True, thermal_cluster).loc[hour, weather_year]
        availabilities.append((price, demand))

    df = pd.DataFrame(data=availabilities, columns=['Price', 'Demand'])
    df = df.sort_values(by='Price')
    df['DemandAccummulated'] = np.cumsum(df.Demand.values[::-1])[::-1]

    fig, ax = plt.subplots()
    df.plot(ax=ax, x='Price', y='DemandAccummulated')
    # ax.set_xlim([0, 500])
    # ax.set_ylim([0, 100000])
    fig.savefig(f'{time_string}_{region}_{commodity}_hour{hour}.png')
    plt.close(fig)


# ------------------------------- #
#            2. Main              #
# ------------------------------- #

@click.group()
def CLI():
    pass

@CLI.command()
def main():
    balmorel_result = 'baf_test_fullyear'
    antares_result = '20250907-2256eco-baf_test_fullyear_ksmooth_area_iter0_y-2050'
    # antares_result = '20250627-1815eco-baf_test_new_fullyear_clsize1000_nr1_y-2050'
    antares_result = '20250908-1240eco-baf_test_fullyear_ksmooth_area_capexincluded_iter0_y-2050'
    antares_result = '20250908-1602eco-baf_test_fullyear_ksmooth_area_capexincluded_alwaysens_iter0_y-2050'

    balmorel_result=MainResults(f"MainResults_{balmorel_result}_Iter0.gdx", paths='Balmorel/base/model')
    antares_result =AntaresOutput(result_name=antares_result)

    for commodity in ['HYDROGEN', 'HEAT']:
        demand_potentials(balmorel_result, antares_result, commodity)
        prices(balmorel_result, antares_result, commodity)

@CLI.command()
def compare():
    balmorel_result = 'baf_test_fullyear'
    antares_result1 = '20250907-2256eco-baf_test_fullyear_ksmooth_area_iter0_y-2050'
    # antares_result = '20250627-1815eco-baf_test_new_fullyear_clsize1000_nr1_y-2050'
    antares_result2 = '20250908-1240eco-baf_test_fullyear_ksmooth_area_capexincluded_iter0_y-2050'
    antares_result2 = ''

    balmorel_result=MainResults(f"MainResults_{balmorel_result}_Iter0.gdx", paths='Balmorel/base/model')
    antares_result1=AntaresOutput(result_name=antares_result1)
    antares_result2=AntaresOutput(result_name=antares_result2)

    # compare_demands(balmorel_result, antares_result1, antares_result2)
    # Get demand response availabilities / supply curves for a day in each season from Antares input
    for season in [168, 168*12, 168*24, 168*36]:
        for region in ['de', 'fr', 'es']:
            for commodity in ['heat', 'hydrogen']:
                for hour in range(season, season+24):    
                    get_availibility_input(region, commodity, hour)


if __name__ == '__main__':
    CLI()
