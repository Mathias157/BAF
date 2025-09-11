"""
TITLE

Description

Created on 14.07.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import pandas as pd
import numpy as np
import os
from pybalmorel import Balmorel
from pybalmorel.utils import symbol_to_df
from configparser import ConfigParser
import click

#%% ------------------------------- ###
###          1. Functions           ###
### ------------------------------- ###

def export_to_generative_model(scenario_folder: str, 
                               weather_year: int, 
                               filename: str,
                               parameters: list = [
                                    'DH_VAR_T',
                                    'DE_VAR_T',
                                    'DH',
                                    'WND_VAR_T', 
                                    'SOLE_VAR_T',
                                    'WTRRRVAR_T',
                                    'WTRRSVAR_S',
                                    'WNDFLH',
                                    'SOLEFLH', 
                                    'WTRRRFLH',
                                    'WTRRSFLH',
                               ],
                               **kwargs):
    
    model = Balmorel('Balmorel', gams_system_directory=kwargs.pop('gams_system_directory', None))
    model.load_incfiles(scenario_folder, overwrite=True)
    
    # Geographic scope
    IR = list(symbol_to_df(model.input_data[scenario_folder], 'IR').RRR)
    IA = list(symbol_to_df(model.input_data[scenario_folder], 'IA').AAA)

    # Verify weather year
    with open(f'Balmorel/{scenario_folder}/data/DH_VAR_T.inc', 'r') as f:
        content = f.read()
        read_weather_year = int(content.split('Weather year ')[1].split('\n')[0])

    if read_weather_year != weather_year:
        raise ValueError(f"Read weather year was {read_weather_year}, but process expect {weather_year}!")

    print('Weather year', weather_year)

    for parameter in parameters:
        print(parameter)
        df = correct_format(symbol_to_df(model.input_data[scenario_folder], parameter), 
                            parameter, 
                            weather_year,
                            IR,
                            IA) 
        
        if parameter == parameters[0]:
            df.T.to_csv(filename%weather_year, header=True)
        else:
            df.T.to_csv(filename%weather_year, mode='a', header=False)
    

def correct_format(df: pd.DataFrame, parameter: str, weather_year: int, IR: list, IA: list):
    
    # Extract unique dimensions
    unique_sets = [col for col in df.columns if col not in ['SSS', 'TTT', 'Value']]
    
    # Remove unnecessary model years
    if 'YYY' in df.columns:
        df = df.query('YYY in ["2030", "2040", "2050"]')
    
    # Remove regions not in geographical scope
    if 'RRR' in df.columns:
        df = df.query(f'RRR in {IR}')
    
    # Remove areas not in geographical scope
    if 'AAA' in df.columns:
        df = df.query(f'AAA in {IA}')
    
    # Add sets
    df.loc[:, ['WY']] = weather_year
    df.loc[:, ['Parameter']] = parameter + "|" + df[unique_sets].astype(str).agg("|".join, axis=1)
    
    S_values = [f'S{i:02d}' for i in range(1, 53)]
    T_values = [f'T{i:03d}' for i in range(1, 169)]
    
    # Find missing values (zero interpreted as no value in GAMS)
    if 'SSS' in df.columns and 'TTT' in df.columns:
        # Fill out some of the missing S and T values
        df = df.pivot_table(index=['WY', 'SSS', 'TTT'], 
                            columns='Parameter',
                            values='Value',
                            fill_value=0)
        
        # Fill out the remaining missing values, if there are any left
        ST_exist = set(df.loc[weather_year].index)
        ST_all = set(pd.MultiIndex.from_product((S_values, T_values)))
        ST_missing = ST_all - ST_exist
        if len(ST_missing) > 0:
            S_missing, T_missing = zip(*ST_missing)
            missing_index = pd.MultiIndex.from_arrays(([weather_year]*len(ST_missing), S_missing, T_missing), names=['WY', 'SSS', 'TTT'])
            df = concat_missing(df, missing_index)
        df = stack_and_reset(df)
    
    elif 'SSS' in df.columns:
        # Fill out some of the missing S values
        df = df.pivot_table(index=['WY', 'SSS'], 
                            columns='Parameter',
                            values='Value',
                            fill_value=0)
        
        # Fill out the remaining missing values
        S_exist = set(df.loc[weather_year].index)
        S_all = set(S_values)
        S_missing = list(S_all - S_exist)
        if len(S_missing) > 0:
            missing_index = pd.MultiIndex.from_arrays(([weather_year]*len(S_missing), S_missing), names=['WY', 'SSS'])
            df = concat_missing(df, missing_index)
        df = stack_and_reset(df)
        
    if not('SSS' in df.columns):
        len_before = len(df)
        df = pd.concat([df]*52, ignore_index=True)
        df.loc[:, ['SSS']] = np.repeat(S_values, len_before)
    if not('TTT' in df.columns):
        len_before = len(df)
        df = pd.concat([df]*168, ignore_index=True)
        df.loc[:, ['TTT']] = np.repeat(T_values, len_before)
    
    sum_before = df.Value.sum()
    df = df.pivot_table(index=['WY', 'SSS', 'TTT'], columns='Parameter', values='Value', aggfunc='sum', fill_value=0)
    sum_after = df.sum().sum()

    if not(np.isclose(sum_before, sum_after, atol=0.1)):
        raise ValueError(f'Aggregation didnt work!\nSum before: {sum_before}\nSum after: {sum_after}')

    return df

def concat_missing(df: pd.DataFrame, missing_index: pd.MultiIndex):
    
    df_missing = pd.DataFrame(
        index=missing_index,
        columns=df.columns,
        data=0.0
    )
    df = pd.concat((df, df_missing))
    
    return df

def stack_and_reset(df: pd.DataFrame):
    df=df.sort_index().stack()
    df.name = 'Value'
    df = df.reset_index()
    return df

def store_config(config: ConfigParser):
    
    with open('Config.ini', 'w') as f:
        config.write(f)

#%% ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###

@click.command()
@click.argument('scenario_folder', type=str)
def main(scenario_folder: str):


    # Load configuration
    config = ConfigParser()
    config.read('Config.ini')
    gams_system_directory = config.get('RunMetaData', 'gams_system_directory')
        
    weather_years = [1982, 1983, 1984, 1985, 1986, 
                    1987, 1988, 1989, 1990, 1991, 
                    1992, 1993, 1994, 1995, 1996, 
                    1997, 1998, 1999, 2000, 2001, 
                    2002, 2003, 2004, 2005, 2006, 
                    2007, 2008, 2009, 2010, 2011, 
                    2012, 2013, 2014, 2015, 2016]
    
    for weather_year in weather_years:
        
        # Change weather year before preprocessing
        config.set('PreProcessing', 'balmorel_weather_year', str(weather_year))
        store_config(config)
        
        # Preprocess data
        os.system('pixi run preprocessing -F --rerun-incomplete')
                
        # Get parameters
        export_to_generative_model(scenario_folder, weather_year, 'Pre-Processing/Output/genmodel_data_WY%d.csv', gams_system_directory=gams_system_directory)


    # Remove profiles that are constant for all weather years, seasons and hours
    df = pd.DataFrame()
    for weather_year in weather_years:
        df = pd.concat((df, pd.read_csv('Pre-Processing/Output/genmodel_data_WY%d.csv'%weather_year, index_col=0, header=[0,1,2]).T))

    constant_idx = (df.iloc[0, :] == df).all()
    df.loc[:, ~constant_idx].to_csv('Pre-Processing/Output/genmodel_input.csv')

if __name__ == '__main__':
    main()
