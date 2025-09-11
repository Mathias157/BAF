"""
TITLE

Description

Created on 15.07.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
import click

#%% ------------------------------- ###
###          1. Functions           ###
### ------------------------------- ###

def load_and_format_supply_curves():
    
    with open('supply_curves.pkl', 'rb') as f:
        f = pkl.load(f)
        
    commodities = f.keys()
    
    # Collect data
    data = []
    for commodity in commodities:
        
        regions = f[commodity].keys()
        
        for region in regions:
            
            # Get parameters in the index
            temp_df = pd.DataFrame(f[commodity][region]).T
            
            # Apply indices
            temp_df['commodity'] = commodity
            temp_df['region'] = region
            temp_df.index.name = 'parameter'
            
            # Append
            data.append(temp_df.reset_index())

    # Get the total dataframe
    df = pd.concat(data)
    
    # Get maximum values (only one point, not actual supply curves, when we do it this way)
    df.loc[:, ['price']] = df['price'].apply(max)
    df.loc[:, ['capacity']] = df['capacity'].apply(max)

    return df

def gaussian_kernel_smooth(x_obs, y_obs, x_new, bandwidth): 
	y_smooth = np.zeros_like(x_new) 
	
	for i, x_target in enumerate(x_new): 
		weights = np.exp(
			-0.5 * ((x_obs - x_target) / bandwidth)**2
		) 
		weights = weights / np.sum(weights) 
		y_smooth[i] = np.sum(weights * y_obs) 
	
	return y_smooth

#%% ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###

@click.command()
def main():

    commodity = 'HEAT'
    region = 'DE'
    output = 'capacity'

    df = (
        load_and_format_supply_curves()
        .query(f'commodity == "{commodity}" and region == "{region}"')
    )
    
    fig, ax = plt.subplots(4)
    
    fig.subplots_adjust(hspace=0.7)
    df.plot(x='parameter', y=output, kind='scatter', ax=ax[0], color='r')
    df.plot(x='parameter', y=output, kind='scatter', ax=ax[1], color='r')
    df.plot(x='parameter', y=output, kind='scatter', ax=ax[2], color='r')
    df.plot(x='parameter', y=output, kind='scatter', ax=ax[3], color='r')
    
    min_param = df.parameter.min()
    max_param = df.parameter.max()
    max_value = df.parameter.abs().max()
    smooth_parameter = np.linspace(min_param-abs(max_value)*0.1,
                                   max_param+abs(max_value)*0.1,
                                   1000)
    
    for i,bandwidth in enumerate([1000, 1e4, 1e5, 1e6]):
        smoothed = gaussian_kernel_smooth(df.parameter, getattr(df, output), smooth_parameter, bandwidth)

        ax[i].plot(smooth_parameter, smoothed)
        ax[i].set_title(f'Bandwidth: {bandwidth}')
    
    ax[3].legend((output, 'smoothed'))
    plt.show()

if __name__ == '__main__':
    main()
