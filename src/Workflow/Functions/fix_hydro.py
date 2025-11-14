"""
Hydro power data input fix

Fixing the fact that run-of-river had incorrect series in Antares, 
which led to incorrect timeseries in Balmorel

Created on 11.11.2025
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
from configparser import ConfigParser

# ------------------------------- #
#          1. Functions           #
# ------------------------------- #

def load_hydro_file(region: str):
    f = pd.read_table(f'Antares/input/hydro/series/{region.lower()}/ror.txt',
                      header=None, sep=',')

    return f

# ------------------------------- #
#            2. Main              #
# ------------------------------- #

@click.command()
def main():
    
    conf = ConfigParser()
    conf.read('Config.ini')
    regions = (
        conf.get('PreProcessing', 'geographical_scope')
        .replace(' ', '')
        .split(',')
    )
    for region in regions:
        df = load_hydro_file(region)
        if len(df) == 26280:
            print(f'Tripled sized ROR series for {region}!')
            df = df.loc[::3]
            df.to_csv(f'Antares/input/hydro/series/{region.lower()}/ror.txt',
                      index=False, header=False, sep='\t')


if __name__ == '__main__':
    main()
