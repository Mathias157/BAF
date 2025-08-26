"""
TITLE

Description

Created on 26.08.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
### ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import click
import pickle
from Functions.GeneralHelperFunctions import AntaresOutput

### ------------------------------- ###
###          1. Functions           ###
### ------------------------------- ###


def get_antares_inadequacy(antares_result: str, regional_mapping: dict):
    AntOut = AntaresOutput(antares_result)

    data = []
    
    for region in regional_mapping.keys():

        ### 1.4 Load Antares Results
        region_result = AntOut.load_area_results(region, temporal='annual')

        ## Unsupplied Energy
        ENS = region_result.loc[0, 'UNSP. ENRG']
        # UNSENR_arr = AntOut.collect_mcyears('UNSP. ENRG', region).quantile(.5, axis=1)   # Hourly median unsupplied energy  

        ## Loss of load expectation 
        LOLE = region_result.loc[0, 'LOLD']

        data.append([region, ENS, LOLE])

    df = pd.DataFrame(data, columns=['Region', 'ENS', 'LOLE'])

    return df
    
### ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###

@click.command()
@click.argument('antares-result')
def main(antares_result: str):

    ## Region mappings
    with open('Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
        A2B_regi = pickle.load(f)

    df = get_antares_inadequacy(antares_result, A2B_regi)

    print(df)
    print(df[['LOLE', 'ENS']].sum())


if __name__ == '__main__':
    main()






