# plot loss function

import pandas as pd
import matplotlib.pyplot as plt
import click
from pybalmorel import MainResults
import sys
sys.path.append('Workflow/Functions')
from GeneralHelperFunctions import get_combined_obj_value
from pathlib import Path

@click.group()
def CLI():
    pass

@CLI.command()
@click.argument('scenario', type=str, required=True)
@click.option('--scenario-folder', type=str, default='operun', help='The scenario folder the scenarios were run in')
def sc(scenario: str, scenario_folder: str):
    
    path = Path(f'Balmorel/{scenario_folder}/model')
    files = [file.name for file in path.glob(f'MainResults_{scenario}*E*.gdx')]
    print(files)
    results = MainResults(files=files,
                          paths=str(path.absolute().resolve()))
    

@CLI.command()
@click.argument('scenario', type=str, required=True)
def adequacy_sc(scenario: str):
    """
    Plot the sum of ENS and LOLE for a specific scenario
    """
        
    df = pd.read_csv('Balmorel/analysis/output/' + scenario + '_adeq.csv')

    loss_values = df['ENS_TWh'] + df['LOLE_h']
    df['loss'] = loss_values
    
    df = df.pivot_table(index='epoch', values='loss', aggfunc='sum')
            
    fig, ax = plt.subplots()
    print(df)
    df.plot(ax=ax)
    ax.set_ylim([0, df.max().max()*1.2])
    fig.savefig('out.png')

@CLI.command()
def adequacy_all():
    """
    Plot the sum of ENS and LOLE for all hardcoded scenarios
    """

    df = pd.DataFrame()    
    for scenario in ['DO_D4W4_dispatch',
                     ]:
        temp = pd.read_csv('Balmorel/analysis/output/' + scenario + '_adeq.csv')

        loss_values = temp['ENS_TWh'] + temp['LOLE_h']
        temp['loss'] = loss_values
        temp['scenario'] = scenario
        
        temp = temp.pivot_table(index='epoch', values='loss', columns='scenario', aggfunc='sum')
        
        # df = df.join(temp)
        df = pd.concat([df, temp], axis=1)

    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_ylabel('Loss Value')
    ax.set_xlabel('Epoch')
    ax.set_ylim([0, df.max().max()*1.2])
    fig.savefig('test.png', transparent=True)

if __name__ == '__main__':
    CLI()