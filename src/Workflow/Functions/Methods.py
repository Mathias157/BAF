"""
Created on 20.04.2024

@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pybalmorel.utils import symbol_to_df
from .GeneralHelperFunctions import store_capcred, AntaresOutput, get_capex

style = 'report'

if style == 'report':
    plt.style.use('default')
    fc = 'white'
elif style == 'ppt':
    plt.style.use('dark_background')
    fc = 'none'

#%% ------------------------------- ###
###        1. Market Value          ###
### ------------------------------- ###

def calculate_elmarket_values(i: int, area: str, BalmArea: str, year: str, 
                            MARKETVAL: str, fMV: pd.DataFrame,
                            AntOut: AntaresOutput):
    
    ## 'Spilled' Energy
    spill = AntOut.collect_mcyears('SPIL. ENRG', area) 
    
    ## Distribute Curtailment
    # Assumming that all spillage is due to renewables            
    total_ren = AntOut.ren_gen['SOLAR PV'].sum(axis=0)
    sol_curt  = (AntOut.ren_gen['SOLAR PV']/total_ren * spill).fillna(0) # This method obtains the correct sum of spilled energy, is that what we want?
    # ax.plot(sol_curt, 'o', markersize=.7, label='Solar Curt. %0.2f TWh\nSolar Total Gen. %0.2f TWh'%(sol_curt.sum() / 1e6, 
    #                                                                                              ren_gen['SOLAR'].sum() / 1e6), color=[.7,.7,0,.5])
    total_ren = AntOut.ren_gen['WIND ONSHORE'].sum(axis=0) 
    wnd_curt  = (AntOut.ren_gen['WIND ONSHORE']/total_ren * spill).fillna(0) # This method obtains the correct sum of spilled energy, is that what we want?
    # ax.plot(wnd_curt, 'o', markersize=.7, label='Wind Curt. %0.2f TWh\nWind Total Gen. %0.2f TWh'%(wnd_curt.sum() / 1e6, 
    #                                                                                              ren_gen['WIND'].sum() / 1e6), color=[0,.7,0,.5])
    # ax.set_ylabel('Curtailment [MWh]')
    # ax.set_xlabel('Hour')
    # ax.set_title(area)
    # ax.legend()
    
    ## Electricity Price
    elprice = AntOut.collect_mcyears('MRG. PRICE', area) 

    
    ## Renewables Markup
    dispatch = (AntOut.ren_gen['WIND ONSHORE'] - wnd_curt)
    wnd_mar = market_value(dispatch, elprice, fMV, 'Wind', year, BalmArea, i)

    dispatch = (AntOut.ren_gen['SOLAR PV'] - sol_curt)
    sol_mar = market_value(dispatch, elprice, fMV, 'Sun', year, BalmArea, i)
    
            
    ## Thermal Generator Markup
    MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s',G)$(GDATA(G,'GDTECHGROUP') EQ WINDTURBINE_OFFSHORE OR GDATA(G,'GDTECHGROUP') EQ WINDTURBINE_ONSHORE)   = %0.2f;\n"%(year, BalmArea, wnd_mar)
    fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : 'Wind', 'Value (eur/MWh)' : wnd_mar}, index=[0])), ignore_index=True)
    MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s',G)$(GDATA(G,'GDTECHGROUP') EQ SOLARPV)   = %0.2f;\n"%(year, BalmArea, sol_mar)
    fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : 'Sun', 'Value (eur/MWh)' : sol_mar}, index=[0])), ignore_index=True)                    

    for tech in AntOut.therm_gen:
        dispatch = AntOut.therm_gen[tech]
        therm_mar = market_value(dispatch, elprice, fMV, tech, year, BalmArea, i)
        MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s',G)$(GDATA(G,'GDFUEL') EQ %s)   = %0.2f;\n"%(year, BalmArea, tech.split('_')[1].upper(), therm_mar)
        fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : tech, 'Value (eur/MWh)' : therm_mar}, index=[0])), ignore_index=True)

    return MARKETVAL, fMV

def create_elmarketvaluestrings(profit_dif: pd.DataFrame,
                                i: int, year: str, BalmArea: str,
                                MARKETVAL: str, fMV: pd.DataFrame):
    
    battery_techs = [
                     'GNR_ES_ELEC_BAT-LITHIO-GRID_E-86_Y-2020',
                     'GNR_ES_ELEC_BAT-LITHIO-GRID_E-86_Y-2030',
                     'GNR_ES_ELEC_BAT-LITHIO-GRID_E-86_Y-2040',
                     'GNR_ES_ELEC_BAT-LITHIO-GRID_E-86_Y-2050',
                     'GNR_ES_ELEC_BAT-LITHIO-PEAK_E-86_Y-2020',
                     'GNR_ES_ELEC_BAT-LITHIO-PEAK_E-86_Y-2030',
                     'GNR_ES_ELEC_BAT-LITHIO-PEAK_E-86_Y-2040',
                     'GNR_ES_ELEC_BAT-LITHIO-PEAK_E-86_Y-2050'
                    ]
    
    for tech in profit_dif.index:
        
        if tech == 'solar pv':
            MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s',G)$(GDATA(G,'GDTECHGROUP') EQ SOLARPV)   = %0.2f;\n"%(year, BalmArea, profit_dif.loc[tech])
            fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : 'Sun', 'Value (eur/MWh)' : profit_dif.loc[tech]}, index=[0])), ignore_index=True)                    
        elif tech == 'wind onshore':
            MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s',G)$(GDATA(G,'GDTECHGROUP') EQ WINDTURBINE_OFFSHORE OR GDATA(G,'GDTECHGROUP') EQ WINDTURBINE_ONSHORE)   = %0.2f;\n"%(year, BalmArea, profit_dif.loc[tech])
            fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : 'Wind', 'Value (eur/MWh)' : profit_dif.loc[tech]}, index=[0])), ignore_index=True)
        elif tech == 'battery':
            for bat_tech in battery_techs:
                MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s','%s') = %0.2f;\n"%(year, BalmArea, bat_tech, profit_dif.loc[tech])
            fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : 'Battery', 'Value (eur/MWh)' : profit_dif.loc[tech]}, index=[0])), ignore_index=True)
        else:
            MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s',G)$(GDATA(G,'GDFUEL') EQ %s)   = %0.2f;\n"%(year, BalmArea, tech.split('_')[1].upper(), profit_dif.loc[tech])
            fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : tech, 'Value (eur/MWh)' : profit_dif.loc[tech]}, index=[0])), ignore_index=True)        
    
    return MARKETVAL, fMV

def calculate_antares_elmarket_profits(i: int, area: str, year: str, 
                            AntOut: AntaresOutput, fAntTechno: pd.DataFrame):
    
    f = AntOut.load_area_results(area)
    
    # Electricity Price
    elprice = f['MRG. PRICE']

    profit_elements = pd.DataFrame(columns=['Revenue', 'CAPEX', 'OPEX', 'Production'])
    
    # Thermal Generators 
    for tech in AntOut.therm_gen:
        production = AntOut.therm_gen[tech]
        profit_elements = calculate_generator_profit_elements(production, elprice, 
                                                    profit_elements, i, year, 
                                                    area, tech, fAntTechno)

    # Renewables (WITH CURTAILMENT)
    for tech in AntOut.ren_gen:
        production = AntOut.ren_gen[tech]
        tech = tech.lower()
        profit_elements = calculate_generator_profit_elements(production, elprice, 
                                                    profit_elements, i, year, 
                                                    area, tech, fAntTechno)
        
    # Batteries
    production = AntOut.load_link_results(['0_battery_turb', area])['FLOW LIN.']
    demand = -AntOut.load_link_results(['0_battery_pmp', area])['FLOW LIN.']
    profit_elements = calculate_generator_profit_elements(production, elprice, profit_elements, i, year,
                                        area, 'battery', fAntTechno)
    if len(production) != 1:
        profit_elements.loc['battery', 'OPEX'] += (demand.values * elprice).sum()
    else:
        pass

    # Calculate profits
    profits_pr_mwh = ((profit_elements['Revenue'] - profit_elements['CAPEX'] - profit_elements['OPEX']) / profit_elements['Production'])
    
    ## Get division by zero NaNs
    idx_nan = profit_elements['Production'] == 0
    profits_pr_mwh.loc[idx_nan] = -profit_elements.loc[idx_nan, 'CAPEX'].values
       
    return profits_pr_mwh

def calculate_antares_h2market_profits(i: int, area: str, year: str, 
                            AntOut: AntaresOutput, fAntTechno: pd.DataFrame):
    f = AntOut.load_area_results(area)
    
    # Electricity Price
    h2price = f['MRG. PRICE']

    profit_elements = pd.DataFrame(columns=['Revenue', 'CAPEX', 'OPEX', 'Production'])
    
    # Thermal Generators 
    for tech in AntOut.therm_gen:
        production = AntOut.therm_gen[tech]
        profit_elements = calculate_generator_profit_elements(production, h2price, 
                                                    profit_elements, i, year, 
                                                    area, tech, fAntTechno)
        
    # Steel Tank Storage
    production = AntOut.load_link_results(['0_h2tank_turb', area])['FLOW LIN.']
    demand = -AntOut.load_link_results(['0_h2tank_pmp', area])['FLOW LIN.']
    profit_elements = calculate_generator_profit_elements(production, h2price, profit_elements, i, year,
                                        area, 'h2 tank', fAntTechno)
    if len(production) != 1:
        profit_elements.loc['h2 tank', 'OPEX'] += (demand.values * h2price).sum()
    else:
        pass
    
    # Large-scale Storage
    production = f['H. STOR']
    demand = -f['H. PUMP']
    profit_elements = calculate_generator_profit_elements(production, h2price, profit_elements, i, year,
                                        area, 'h2 cavern', fAntTechno)
    if len(production) != 1:
        profit_elements.loc['h2 cavern', 'OPEX'] += (demand.values * h2price).sum()
    else:
        pass
    
    # Calculate profits
    profits_pr_mwh = ((profit_elements['Revenue'] - profit_elements['CAPEX'] - profit_elements['OPEX']) / profit_elements['Production'])
    
    ## Get division by zero NaNs
    idx_nan = profit_elements['Production'] == 0
    profits_pr_mwh.loc[idx_nan] = -profit_elements.loc[idx_nan, 'CAPEX'].values
       
    return profits_pr_mwh


def create_h2marketvaluestrings(profit_dif: pd.DataFrame,
                                i: int, year: str, BalmArea: str,
                                MARKETVAL: str, fMV: pd.DataFrame):
    
    h2tank_techs = [
                    'GNR_H2S_H2-TNKC_Y-2020',
                    'GNR_H2S_H2-TNKC_Y-2030',
                    'GNR_H2S_H2-TNKC_Y-2040',
                    'GNR_H2S_H2-TNKC_Y-2050'
                    ]
    
    h2cavern_techs = [
        'GNR_H2S_H2-CAVERN_Y-2020',
        'GNR_H2S_H2-CAVERN_Y-2030',
        'GNR_H2S_H2-CAVERN_Y-2040',
        'GNR_H2S_H2-CAVERN_Y-2050'
    ]
    
    for tech in profit_dif.index:
        
        if tech == 'h2 tank':
            for h2tank_tech in h2tank_techs:
                MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s','%s') = %0.2f;\n"%(year, BalmArea, h2tank_tech, profit_dif.loc[tech])
            fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : 'H2 Tank', 'Value (eur/MWh)' : profit_dif.loc[tech]}, index=[0])), ignore_index=True)
        elif tech == 'h2 cavern':
            for h2cavern_tech in h2cavern_techs:
                MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s','%s') = %0.2f;\n"%(year, BalmArea, h2cavern_tech, profit_dif.loc[tech])
            fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : 'H2 Cavern', 'Value (eur/MWh)' : profit_dif.loc[tech]}, index=[0])), ignore_index=True)
        elif tech == 'steam-methane-reforming_natgas':
            MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s','GNR_STEAM-REFORMING_E-70_Y-2020')   = %0.2f;\n"%(year, BalmArea, profit_dif.loc[tech])
            fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : tech, 'Value (eur/MWh)' : profit_dif.loc[tech]}, index=[0])), ignore_index=True)        
        elif tech == 'steam-methane-reforming-ccs_natgas': 
            MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s','GNR_STEAM-REFORMING-CCS_E-70_Y-2020')   = %0.2f;\n"%(year, BalmArea, profit_dif.loc[tech])
            fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : tech, 'Value (eur/MWh)' : profit_dif.loc[tech]}, index=[0])), ignore_index=True)        
    
    return MARKETVAL, fMV
    
    
def make_antares_names(df: pd.DataFrame):
    
    # Create most of the names
    series = df['Tech'].str.lower() + '_' + df['F'].str.lower()
    
    # Rename unique's
    series = series.str.replace('electrolyzer_electric', 'electrolyser')
    series = series.str.replace('solar-pv_sun', 'solar pv')
    series = series.str.replace('wind-on_wind', 'wind onshore')
    series = series.str.replace('wind-off_wind', 'wind onshore')
    
    ## Battery
    idx = (df.Tech == 'INTRASEASONAL-ELECT-STORAGE') & (df.G.str.find('BAT-LITHIO') != -1)
    series[idx] = 'battery'
    
    ## Steam reforming CCS
    idx = (df.G.str.find('CCS') != -1) & (df.Tech.str.find('STEAMREFORMING') != -1)
    series[idx] = 'steamreforming-ccs_natgas'
    
    ## Hydrogen Tank Storage
    idx = (df.Tech == 'H2-STORAGE') & (df.G.str.find('TNKC') != -1)
    series[idx] = 'h2 tank'
    
    ## Hydrogen Cavern Storage
    idx = (df.Tech == 'H2-STORAGE') & (df.G.str.find('CAVERN') != -1)
    series[idx] = 'h2 cavern'
    
    return series

def calculate_generator_profit_elements(production: tuple[pd.DataFrame, np.array], 
                              market_price: pd.DataFrame,
                              profit_elements: pd.DataFrame,
                              i: int,
                              year: str,
                              area: str,
                              tech: str,
                              fAntTechno: pd.DataFrame):
    
    if len(production) != 1:
        try:
            production = production.values[:, 0]
        except IndexError:
            production = production.values
        
        revenue = (production * market_price).sum()
        profit_elements.loc[tech, 'Revenue'] = revenue
        profit_elements.loc[tech, 'Production'] = production.sum()
        profit_elements.loc[tech, 'CAPEX'] = fAntTechno.loc[(i, year, area, tech), 'CAPEX']
        profit_elements.loc[tech, 'OPEX'] = production.sum() * fAntTechno.loc[(i, year, area, tech), 'OPEX']
    else:
        profit_elements.loc[tech, 'Revenue'] = 0
        profit_elements.loc[tech, 'Production'] = 0
        profit_elements.loc[tech, 'CAPEX'] = fAntTechno.loc[(i, year, area, tech), 'CAPEX']
        profit_elements.loc[tech, 'OPEX'] = fAntTechno.loc[(i, year, area, tech), 'OPEX']
    
    return profit_elements

def calculate_h2market_values(i: int, area: str, BalmArea: str, year: str, 
                            MARKETVAL: str, fMV: pd.DataFrame,
                            AntOut: AntaresOutput):
    
    
    ## Hydrogen Price
    h2price = AntOut.collect_mcyears('MRG. PRICE', area) 
    
    ## Electrolyser Markup
    dispatch = AntOut.collect_mcyears('FLOW LIN.', ['x_c3', area])
    elec_mar = market_value(dispatch, h2price, fMV, 'Electrolyser', year, BalmArea, i)
    MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s',G)$(GDATA(G,'GDTECHGROUP') EQ HYDROGEN_GETOHH2 OR GDATA(G,'GDTECHGROUP') EQ HYDROGEN_GETOH2) = %0.2f;\n"%(year, BalmArea, elec_mar)
    fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : 'Electrolyser', 'Value (eur/MWh)' : elec_mar}, index=[0])), ignore_index=True)
            
    ## Thermal Generator Markup
    for tech in AntOut.therm_gen:
        if 'ccs' in tech:
            name = 'GNR_STEAM-REFORMING-CCS_E-70_Y-2020'
        else:
            name = 'GNR_STEAM-REFORMING_E-70_Y-2020'
        
        dispatch = AntOut.therm_gen[tech]
        therm_mar = market_value(dispatch, h2price, fMV, tech, year, BalmArea, i)
        MARKETVAL = MARKETVAL + "ANTBALM_MARKETVAL('%s','%s','%s') = %0.2f;\n"%(year, BalmArea, name, therm_mar)
        fMV = pd.concat((fMV, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : BalmArea, 'Tech' : tech, 'Value (eur/MWh)' : therm_mar}, index=[0])), ignore_index=True)

    return MARKETVAL, fMV

def market_value(dispatch: pd.DataFrame, market_price: pd.DataFrame,
                 previous_market_values: pd.DataFrame, 
                 tech: str, year: str, BalmArea: str, i: int):
    
    # If there is generation
    if dispatch.values.sum() > 1e-6:
        markup = ((dispatch*market_price).sum() / dispatch.sum() - market_price.mean()).mean()          
    
    else:
        try:
            # Use from earlier iteration if no markup
            idx = (previous_market_values.Iter == i - 1) & (previous_market_values.Year == int(year)) &\
                (previous_market_values.Region == BalmArea) & (previous_market_values.Tech == tech)
            markup = previous_market_values.loc[idx, 'Value (eur/MWh)'].values[0]
        
        except IndexError:
            # Set to zero if no earlier markup
            markup = 0
    
    return round(markup, 4)


#%% ------------------------------- ###
###       2. Capacity Credit        ###
### ------------------------------- ###


def calculate_generator_capacity_credits(i: int, BalmArea: str, area: str, year: str,
                            AntOut: AntaresOutput,
                            cap: pd.DataFrame, 
                            A2B_regi: dict, CC: pd.DataFrame,
                            update_thermal: bool,
                            CAPCRED_G: str):
    
    
    load = AntOut.load

    
    # Get capacity corresponding to Antares region - (Same capacity credit in all Balmorel regions)
    idx = cap.R != cap.R
    for ant_area in A2B_regi[area]:
        idx = idx | (cap.R == ant_area)
    
    for tech in AntOut.therm_gen.keys():
        # If thermal capacity credits should not be updated
        if not(update_thermal) and i != 0:
            # Use capacity credit from previous iteration
            val = 1
            tech_cap = 0 # A tech cap of zero will ignore the value and use earlier iterations or set val to 1
        
        # If there's any thermal generation and capacity credit should be updated
        elif AntOut.therm_gen != 0:
            # Get capacity
            tech_cap = np.sum(cap.loc[idx & (cap.F == tech.split('_')[1].upper()) &\
                                    (cap.Tech == tech.split('_')[0].upper()) & (cap.Y == year), 'Value'])*1e3
            
            # If there's any generation from this tech
            if AntOut.therm_gen[tech].values.sum() > 1e-6:
                
                ### 1.6 Capacity Credit
                # Calculate residual demand corresponding to technology generation
                prod = AntOut.therm_gen[tech]

                val = capacity_credit_value(load, prod, tech_cap, 'max_hours')
            
            # If it exist but is never used, the capacity credit is zero (a tech cap of zero will ignore that and use earlier iterations or set it to 1)
            else:
                val = 0
            
        # Else, at zero capacity store_capcred will use previous iteration
        else:
            val = 1 # Will be ignored due to 0 tech cap in store_capcred, but need input
            tech_cap = 0
        
        # Save capacity credit
        CC = store_capcred(CC, i, year, BalmArea, tech, tech_cap, val)

        # Get CHP type
        CHP_Type = tech.split('_')[0].replace('-ccs', '').replace('condensing', 'GCND').replace('chp-back-pressure', 'GBPR').replace('chp-extraction', 'GEXT').replace('fuelcell', 'HYDROGEN_GH2TOE')
            
        # Update thermal only in the first round, if choice not update thermal
        CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s',G)$(GDATA(G,'GDFUEL') EQ %s AND GDATA(G, 'GDTYPE') EQ %s)   = %0.5f;\n"%(year, BalmArea, 
                                                                                                                                tech.split('_')[1].upper(),
                                                                                                                                CHP_Type,
                                                                                                                                CC.loc[(i, year, BalmArea), tech]+0.00001) # Add 0.00001 to ensure feasibility
        
    for tech in AntOut.ren_gen:
        # Calculate residual demand corresponding to technology generation
        prod = AntOut.ren_gen[tech]
        
        tech_cap = np.sum(cap.loc[idx & (cap.F == tech.replace('SOLAR PV', 'SUN').replace('WIND ONSHORE', 'WIND')) &\
                                        (cap.Commodity == 'ELECTRICITY') & (cap.Y == year), 'Value'])*1e3
        
        if tech_cap > 1e-6:
            
            val = capacity_credit_value(load, prod, tech_cap, 'max_hours')

        else:
            val = 1 # Just a placeholder, store_capcred will use the value from last iteration or set CC as 1
                
        CC = store_capcred(CC, i, year, BalmArea, tech, tech_cap, val)
        if tech == 'WIND ONSHORE':
            CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s',G)$(GDATA(G,'GDTECHGROUP') EQ WINDTURBINE_OFFSHORE OR GDATA(G,'GDTECHGROUP') EQ WINDTURBINE_ONSHORE) = %0.5f;\n"%(year, BalmArea, CC.loc[(i, year, BalmArea), tech]+0.00001)
        else:                                                                                        
            CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s',G)$(GDATA(G,'GDTECHGROUP') EQ SOLARPV)   = %0.5f;\n"%(year, BalmArea, CC.loc[(i, year, BalmArea), tech]+0.00001)
    
    # Battery storage
    tech_cap = np.sum(cap.loc[idx & (cap.Tech == 'INTRASEASONAL-ELECT-STORAGE') & (cap.G.str.find('BAT-LITHIO') != -1) & (cap.Y == year), 'Value'])*1e3
    if tech_cap > 1e-6:
        # Load discharge
        prod = AntOut.collect_mcyears('FLOW LIN.', ['0_battery_turb', area])

        val = capacity_credit_value(load, prod, tech_cap, 'max_hours')
    else:
        val = 1 # Just a placeholder, store_capcred will use the value from last iteration or set CC as 1

    CC = store_capcred(CC, i, year, BalmArea, 'Li-Ion', tech_cap, val)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s',G)$(GDATA(G, 'GDTECHGROUP') EQ ELECTRICITY_BATTERY) = %0.5f;\n"%(year, BalmArea, CC.loc[(i, year, BalmArea), 'Li-Ion']+0.00001)

    return CC, CAPCRED_G

def calculate_h2generator_capacity_credits(i: int, BalmArea: str, area: str, year: str,
                                AntOut: AntaresOutput,
                                cap: pd.DataFrame, 
                                A2B_regi_h2: dict, CCH2: pd.DataFrame,
                                update_thermal: bool,
                                CAPCRED_G: str):
    
    
    load = AntOut.load

    
    # Get capacity corresponding to Antares region - (Same capacity credit in all Balmorel regions)
    idx = cap.R != cap.R
    for ant_area in A2B_regi_h2[area]:
        idx = idx | (cap.R == ant_area)
    
    for tech in AntOut.therm_gen.keys():
        if 'ccs' in tech:
            name = 'GNR_STEAM-REFORMING-CCS_E-70_Y-2020'
        else:
            name = 'GNR_STEAM-REFORMING_E-70_Y-2020'
        
        # If thermal capacity credits should not be updated
        if not(update_thermal) and i != 0:
            # Use capacity credit from previous iteration
            val = 1
            tech_cap = 0 # A tech cap of zero will ignore the value and use earlier iterations or set val to 1
        
        # If there's any thermal generation and capacity credit should be updated
        elif AntOut.therm_gen != 0:
            # Get capacity
            tech_cap = np.sum(cap.loc[idx & (cap.G == name) & (cap.Y == year), 'Value'])*1e3
            
            # If there's any generation from this tech
            if AntOut.therm_gen[tech].values.sum() > 1e-6:
                
                ### 1.6 Capacity Credit
                # Calculate residual demand corresponding to technology generation
                prod = AntOut.therm_gen[tech]

                val = capacity_credit_value(load, prod, tech_cap, 'mean_all')
            
            # If it exist but is never used, the capacity credit is zero (a tech cap of zero will ignore that and use earlier iterations or set it to 1)
            else:
                val = 0
            
        # Else, at zero capacity store_capcred will use previous iteration
        else:
            val = 1 # Will be ignored due to 0 tech cap in store_capcred, but need input
            tech_cap = 0
        
        # Save capacity credit
        CCH2 = store_capcred(CCH2, i, year, BalmArea, tech, tech_cap, val)


        # Update thermal only in the first round, if choice not update thermal
        CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s', '%s')   = %0.5f;\n"%(year, BalmArea, 
                                                                                    name,
                                                                                    CCH2.loc[(i, year, BalmArea), tech]+0.00001)
    
    # Electrolyser
    tech_cap = np.sum(cap.loc[idx & (cap.Tech == 'ELECTROLYZER') & (cap.Y == year), 'Value'])*1e3
    if tech_cap > 1e-6:
        # Load discharge
        prod = AntOut.collect_mcyears('FLOW LIN.', ['x_c3', area])

        val = capacity_credit_value(load, prod, tech_cap, 'mean_all')
    else:
        val = 1 # Just a placeholder, store_capcred will use the value from last iteration or set CCH2 as 1
    CCH2 = store_capcred(CCH2, i, year, BalmArea, 'Electrolyser', tech_cap, val)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s', G)$(GDATA(G,'GDTECHGROUP') EQ ELECTROLYZER) = %0.5f;\n"%(year, BalmArea, CCH2.loc[(i, year, BalmArea), 'Electrolyser']+0.00001)

    # Tank storage
    tech_cap = np.sum(cap.loc[idx & (cap.Tech == 'H2-STORAGE') & (cap.G.str.find('H2-TNKC') != -1) & (cap.Y == year), 'Value'])*1e3
    if tech_cap > 1e-6:
        # Load discharge
        prod = AntOut.collect_mcyears('FLOW LIN.', ['0_h2tank_turb', area])

        val = capacity_credit_value(load, prod, tech_cap, 'mean_all')
    else:
        val = 1 # Just a placeholder, store_capcred will use the value from last iteration or set CCH2 as 1

    CCH2 = store_capcred(CCH2, i, year, BalmArea, 'H2Tank', tech_cap, val)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s','GNR_H2S_H2-TNKC_Y-2020') = %0.5f;\n"%(year, BalmArea, CCH2.loc[(i, year, BalmArea), 'H2Tank']+0.00001)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s','GNR_H2S_H2-TNKC_Y-2030') = %0.5f;\n"%(year, BalmArea, CCH2.loc[(i, year, BalmArea), 'H2Tank']+0.00001)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s','GNR_H2S_H2-TNKC_Y-2040') = %0.5f;\n"%(year, BalmArea, CCH2.loc[(i, year, BalmArea), 'H2Tank']+0.00001)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s','GNR_H2S_H2-TNKC_Y-2050') = %0.5f;\n"%(year, BalmArea, CCH2.loc[(i, year, BalmArea), 'H2Tank']+0.00001)

    # Cavern storage
    tech_cap = np.sum(cap.loc[idx & (cap.Tech == 'H2-STORAGE') & (cap.G.str.find('H2-CAVERN') != -1) & (cap.Y == year), 'Value'])*1e3
    if tech_cap > 1e-6:
        # Load discharge
        prod = AntOut.collect_mcyears('H. STOR', area)

        val = capacity_credit_value(load, prod, tech_cap, 'mean_all')
    else:
        val = 1 # Just a placeholder, store_capcred will use the value from last iteration or set CCH2 as 1

    CCH2 = store_capcred(CCH2, i, year, BalmArea, 'H2Cavern', tech_cap, val)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s','GNR_H2S_H2-CAVERN_Y-2020') = %0.5f;\n"%(year, BalmArea, CCH2.loc[(i, year, BalmArea), 'H2Cavern']+0.00001)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s','GNR_H2S_H2-CAVERN_Y-2030') = %0.5f;\n"%(year, BalmArea, CCH2.loc[(i, year, BalmArea), 'H2Cavern']+0.00001)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s','GNR_H2S_H2-CAVERN_Y-2040') = %0.5f;\n"%(year, BalmArea, CCH2.loc[(i, year, BalmArea), 'H2Cavern']+0.00001)
    CAPCRED_G = CAPCRED_G + "ANTBALM_GCAPCRED('%s', '%s','GNR_H2S_H2-CAVERN_Y-2050') = %0.5f;\n"%(year, BalmArea, CCH2.loc[(i, year, BalmArea), 'H2Cavern']+0.00001)


    return CCH2, CAPCRED_G


def calculate_link_capacity_credits(i: int, area: str, year: str, AntOut: AntaresOutput, 
                            tran: pd.DataFrame, CC: pd.DataFrame,
                            A2B_regi: dict, CAPCRED_X: str, method: str = 'max_hours',
                            carrier='X'):
    
    load = AntOut.load
    
    ## Get capacity credit of links
    for area2 in A2B_regi.keys():
        
        try:
            # Link results only defined once in Antares, try loading this direction...
            A_to_prod =  AntOut.collect_mcyears('FLOW LIN.', [area2, area])
                
        except FileNotFoundError: 
            try:   
                # ...if it doesn't exist, then it's defined in the other direction
                A_to_prod =  -AntOut.collect_mcyears('FLOW LIN.', [area, area2])
            except FileNotFoundError:
                # No link in this direction
                continue
        
        # Only look at importing values
        idx = A_to_prod.values < 0
        A_to_prod.loc[idx] = 0
        
        # Loop through Balmorel areas to see which capacities exist, then assign capacity credit
        for BalmArea1 in A2B_regi[area]:
            for BalmArea2 in A2B_regi[area2]:
                tech_cap = tran.loc[(tran.RI == BalmArea1) & (tran.RE == BalmArea2) & (tran.Y == year), 'Value'].sum() * 1e3
                
                if tech_cap > 1e-6:
                    # print(tech_cap)
                    to_cap_cred = capacity_credit_value(load, A_to_prod, tech_cap, method)
                else:
                    to_cap_cred = 0
                
                # NOTE: Some non-existing investment options will get capcred 1 - this is taken care of by hardcoded zero-statements before saving the .inc file
                CC = store_capcred(CC, i, year, BalmArea1, BalmArea2, tech_cap, to_cap_cred)
                CAPCRED_X = CAPCRED_X + "ANTBALM_%sCAPCRED('%s', '%s','%s') = %0.5f;\n"%(carrier, year, BalmArea1, BalmArea2, CC.loc[(i, year, BalmArea1), BalmArea2]+0.00001)
             
    return CC, CAPCRED_X


def capacity_credit_value(load: pd.DataFrame, flow: pd.DataFrame,
                          tech_cap: float, method: str):
        val = pd.DataFrame({}) 
        for col in load.columns:
            if method == 'max_hours':
                # Find max hours
                net_load = load[col] - flow[col]
                max_hours = net_load.sort_values().iloc[-100:].index # the 100 hours with the highest net load
                # Save capacity credit
                val[col] = (flow.loc[max_hours, col] / tech_cap).values
            elif method == 'mean_all':
                val[col] = (flow.loc[:, col]         / tech_cap).values
                

        return val.mean().mean()


def recalculate_resmar(db, 
                       year: int, BalmArea: str,  carrier: str,
                       heggarty_func: str, interpolation_data: pd.DataFrame,
                       RESMAR: pd.DataFrame, LOLD: float):
    
    if carrier.lower() == 'hydrogen':
        old_resmar = symbol_to_df(db, 'ANTBALM_H2RESMAR', ['Y', 'R', 'Value']).groupby(by=['Y', 'R']).aggregate({'Value' : np.sum})
    elif carrier.lower() == 'electricity':
        old_resmar = symbol_to_df(db, 'ANTBALM_RESMAR', ['Y', 'R', 'Value']).groupby(by=['Y', 'R']).aggregate({'Value' : np.sum})
        
    # Linear fit parameters 
    if heggarty_func.lower() == 'conservative':
        params = [0.08005213, 0.00099886]
    elif heggarty_func.lower() == 'risky':
        params = [0.010181  , 0.00247613]

    # Check if there is a capacity constraint for the specific year and region
    try:
        # print('Calculate new reserve margin for %s in %s for %s'%(BalmArea, year, carrier))
        # Convert LOLD to reserve margin
        RESMAR.loc[int(year), BalmArea] = adjust_resmar(old_resmar.loc[(str(year), BalmArea), 'Value'], 
                                                        LOLD, interpolation_data, params)
        
    except KeyError:
        # If there is not, check if LOLD is above 3h
        if RESMAR.loc[int(year), BalmArea] > 3:
            # If it is, make reserve margin = 5%
            RESMAR.loc[int(year), BalmArea] = 0.05
        else:
            RESMAR.loc[int(year), BalmArea] = 0
            
        # print('There was no reserve margin for %s in %s, setting RESMAR to %0.2f, because of a LOLD of %0.2f'%(carrier, BalmArea, 
        #                                                                                                        RESMAR.loc[int(year), BalmArea],
        #                                                                                                        LOLD))
        
    return RESMAR


def adjust_resmar(old_resmar, LOLD, interpolate_data, params):
    # Adjust reserve margin function 
    # Based on reading of Heggarty function
    # if LOLD < 2:
    #     resmar = old_resmar - 2/75 * (LOLD - 2)**2
    # print('Old reserve margin:\t', old_resmar)
    # print('Loss of load durat:\t', LOLD)
    if LOLD > 10:
        # Linear curve fitted on  
        resmar = old_resmar + params[1] * (LOLD - 2) + params[0]
    else:
        resmar = old_resmar + np.interp(LOLD, interpolate_data[0], 
                                        interpolate_data[1])
        
    # If reserve margin is negative, then delete
    if resmar < 0:
        resmar = 0    
    
    # print('New reserve margin:\t', resmar)
    
    return resmar


#%% ------------------------------- ###
###        3. Fictive Demand        ###
### ------------------------------- ###

# A function that increases from -0.9 at x = 0 to 0 at x = 2.5
decrease_function = lambda x: 0.9/2.5*x - 0.9

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


def calculate_elfictdem(FICTDEMALLOC: str, 
                        balm_t: pd.DataFrame, t: pd.DataFrame,
                        BalmArea: str, weight: float,
                        year: str,
                        fict_de_factor: str,
                        fDEVAR: pd.DataFrame,
                        UNSENR_arr: pd.DataFrame, 
                        FICTDE: str, LOLD: float,
                        ENS: pd.DataFrame, i: int, 
                        B2A_regi: dict,
                        negative_feedback: bool = False):
    if FICTDEMALLOC == 'lole_ts':       
        
        ### 1.7 Fictive Demand 
        front_t = t.copy().drop(columns=['S', 'T'])
        front_t.index = np.arange(-len(t), 0)
        back_t = t.copy().drop(columns=['S', 'T'])
        back_t.index = np.arange(len(t), 2*len(t))
        temp_t = pd.concat((front_t, t.copy(), back_t))
        for n,row in balm_t.iterrows():
            # Get current hour
            h1 = temp_t.loc[(temp_t['S'] == row['S']) & (temp_t['T'] == row['T']), 'Hour'].values[0]
            
            # Get previous hour
            try:
                h0 = temp_t.loc[(temp_t['S'] == balm_t.loc[n-1, 'S']) & (temp_t['T'] == balm_t.loc[n-1, 'T']), 'Hour'].values[0]
                dif0 = round((h1 - h0)/2)
            except KeyError:
                h0 = temp_t.loc[(temp_t['S'] == balm_t.loc[len(balm_t)-1, 'S']) & (temp_t['T'] == balm_t.loc[len(balm_t)-1, 'T']), 'Hour'].values[0]
                dif0 = round((h1 + len(t) - h0)/2)
            
            # Get last hour
            try:
                h2 = temp_t.loc[(temp_t['S'] == balm_t.loc[n+1, 'S']) & (temp_t['T'] == balm_t.loc[n+1, 'T']), 'Hour'].values[0]
                dif2 = round((h2 - h1)/2)
            except KeyError:
                h2 = temp_t.loc[(temp_t['S'] == balm_t.loc[0, 'S']) & (temp_t['T'] == balm_t.loc[0, 'T']), 'Hour'].values[0]
                dif2 = round((h2 + len(t) - h1)/2)
            
            # Accumulated Unsupplied Energy
            idx = (temp_t.index >= h1 - dif0) & (temp_t.index < h1 + dif2 - 1)
            balm_t.loc[n, BalmArea + '_UNSELEC'] = temp_t.loc[idx, 'UNSENR'].sum()
        
        agg = balm_t.groupby(by=['S'])
        agg = agg.aggregate({BalmArea + '_UNSELEC' : np.sum})

        if LOLD > 3:
            fDEVAR.loc[(year, BalmArea), list(agg.index)] += np.array(weight*agg[BalmArea + '_UNSELEC']*eval(fict_de_factor))
        else:
            pass
        
        # Store for Balmorel output
        FICTDE = FICTDE + "DE('%s','%s','FICTIVE_%s') = %0.2f;\n"%(year, BalmArea, year, fDEVAR.loc[year, BalmArea].sum()) # <--- Save this in a list or array instead, will accumulate el-demand from electrolyser as well

    elif FICTDEMALLOC == 'existing_ts': 
            
        if LOLD > 3:
            fDEVAR.loc[(year, BalmArea), :] += weight*UNSENR_arr.sum()*eval(fict_de_factor) / len(fDEVAR.columns)
            print('%s adding elfictdem: '%BalmArea, weight*UNSENR_arr.sum()*eval(fict_de_factor))
        elif (LOLD < 2.5) and negative_feedback and (i != 0):
            # Subtract 50% of ENS from last iteration
            last_ENS = ENS.pivot_table(index=['Iter', 'Year', 'Region'],
                            values='Value (MWh)').loc[(i-1, int(year), B2A_regi[BalmArea][0]), 'Value (MWh)']
            previous_factor = eval(fict_de_factor.replace('i', '(i-1)'))
            # print('Last ENS: ')
            # print((last_ENS*previous_factor).to_string())
            subtraction = weight*last_ENS*decrease_function(LOLD)*previous_factor / len(fDEVAR.columns)
            print('%s subtracting elfictdem: '%BalmArea, subtraction*len(fDEVAR.columns))
            fDEVAR.loc[(year, BalmArea), :] += float(subtraction) # decrease_function is negative, so we are adding a negative number
            
            # Make sure it's not negative (can happen if there was a small ENS but no fictdem added because LOLE is < 3 h)
            if np.all(fDEVAR.loc[(year, BalmArea), :] < 0):
                fDEVAR.loc[(year, BalmArea), :] = 0 
        else:
            print('Didnt add FICTDE for %s %s because EL LOLD %0.2f'%(year, BalmArea, LOLD))
            pass
        
        # Annual demand
        FICTDE = FICTDE + "DE('%s','%s','FICTIVE_%s') = %0.2f;\n"%(year, BalmArea, year, fDEVAR.loc[(year, BalmArea), :].sum()) # <--- Save this in a list or array instead, will accumulate el-demand from electrolyser as well

    return fDEVAR, FICTDE

def calculate_h2fictdem(FICTDEMALLOC: str, 
                        balm_t: pd.DataFrame, t: pd.DataFrame,
                        BalmArea: str, area: str, year: str,
                        A2B_regi_h2: dict, A2B_DH2_weights: dict, 
                        fict_dh2_factor: str,
                        fDH2VAR: pd.DataFrame,
                        UNSENR_arr: pd.DataFrame,
                        FICTDH2: str, LOLD: float,
                        ENS: pd.DataFrame, i: int,
                        B2A_regi_h2: dict, negative_feedback: bool = False):
    
    if FICTDEMALLOC == 'lole_ts':  
        front_t = t.copy().drop(columns=['S', 'T'])
        front_t.index = np.arange(-len(t), 0)
        back_t = t.copy().drop(columns=['S', 'T'])
        back_t.index = np.arange(len(t), 2*len(t))
        temp_t = pd.concat((front_t, t.copy(), back_t))
        
        
        for BalmArea in A2B_regi_h2[area]:
            # Weight
            weight = A2B_DH2_weights[area][BalmArea]
        
            for n,row in balm_t.iterrows():
                # Get current hour
                h1 = temp_t.loc[(temp_t['S'] == row['S']) & (temp_t['T'] == row['T']), 'Hour'].values[0]
                
                # Get previous hour
                try:
                    h0 = temp_t.loc[(temp_t['S'] == balm_t.loc[n-1, 'S']) & (temp_t['T'] == balm_t.loc[n-1, 'T']), 'Hour'].values[0]
                    dif0 = round((h1 - h0)/2)
                except KeyError:
                    h0 = temp_t.loc[(temp_t['S'] == balm_t.loc[len(balm_t)-1, 'S']) & (temp_t['T'] == balm_t.loc[len(balm_t)-1, 'T']), 'Hour'].values[0]
                    dif0 = round((h1 + len(t) - h0)/2)
                
                # Get last hour
                try:
                    h2 = temp_t.loc[(temp_t['S'] == balm_t.loc[n+1, 'S']) & (temp_t['T'] == balm_t.loc[n+1, 'T']), 'Hour'].values[0]
                    dif2 = round((h2 - h1)/2)
                except KeyError:
                    h2 = temp_t.loc[(temp_t['S'] == balm_t.loc[0, 'S']) & (temp_t['T'] == balm_t.loc[0, 'T']), 'Hour'].values[0]
                    dif2 = round((h2 + len(t) - h1)/2)
                
                # Accumulated Unsupplied Energy
                idx = (temp_t.index >= h1 - dif0) & (temp_t.index < h1 + dif2 - 1)
                balm_t.loc[n, BalmArea + '_UNSH2'] = temp_t.loc[idx, 'UNSENR'].sum()
            
            agg = balm_t.groupby(by=['S'])
            agg = agg.aggregate({BalmArea + '_UNSH2' : np.sum})
            
            if LOLD > 3:       
                fDH2VAR.loc[(year, BalmArea), list(agg.index)] += np.array(weight*agg[BalmArea + '_UNSH2']*eval(fict_dh2_factor))
            else:
                pass
            
            FICTDH2 = FICTDH2 + "HYDROGEN_DH2('%s','%s') = HYDROGEN_DH2('%s','%s') + %0.2f;\n"%(year, BalmArea, year, BalmArea, fDH2VAR.loc[year, BalmArea].sum()) # <--- Save this in a list or array instead, will accumulate el-demand from electrolyser as well

    elif FICTDEMALLOC == 'existing_ts': 
        
        for BalmArea in A2B_regi_h2[area]:
            # Weight
            weight = A2B_DH2_weights[area][BalmArea] 
            
            if LOLD > 3:
                fDH2VAR.loc[(year, BalmArea), :] += weight*UNSENR_arr.sum()*eval(fict_dh2_factor) / len(fDH2VAR.columns)
                print('%s adding h2fictdem: '%BalmArea, weight*UNSENR_arr.sum()*eval(fict_dh2_factor))
            elif LOLD < 2 and negative_feedback and i != 0:
                # Subtract 50% of ENS from last iteration
                last_ENS = ENS.pivot_table(index=['Iter', 'Year', 'Region'],
                                values='Value (MWh)').loc[(i-1, int(year), B2A_regi_h2[BalmArea][0]), 'Value (MWh)']
                previous_factor = eval(fict_dh2_factor.replace('i', '(i-1)'))
                subtraction = weight*last_ENS*decrease_function(LOLD)*previous_factor / len(fDH2VAR.columns)
                print('%s subtracting h2fictdem: '%BalmArea, subtraction * len(fDH2VAR.columns))
                fDH2VAR.loc[(year, BalmArea), :] += subtraction # Adding a negative number (decrease_function)
                
                # Make sure it's not negative (can happen if there was a small ENS but no fictdem added because LOLE is < 3 h)
                if np.all(fDH2VAR.loc[(year, BalmArea), :] < 0):
                    fDH2VAR.loc[(year, BalmArea), :] = 0 
            else:
                print('Didnt add FICTDH2 for %s %s because H2 LOLD %0.2f'%(year, BalmArea, LOLD))
                pass

            # Annual demand
            FICTDH2 = FICTDH2 + "HYDROGEN_DH2('%s','%s') = HYDROGEN_DH2('%s','%s') + %0.2f;\n"%(year, BalmArea, year, BalmArea, fDH2VAR.loc[(year, BalmArea), :].sum()) # <--- Save this in a list or array instead, will accumulate el-demand from electrolyser as well

    return fDH2VAR, FICTDH2
