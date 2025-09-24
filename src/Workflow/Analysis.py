### ------------------------------- ###
###            0. Import            ###
### ------------------------------- ###

import gams
import gams.control
import pandas as pd
import numpy as np
from datetime import datetime
import platform
OS = platform.platform().split('-')[0]
import matplotlib.pyplot as plt
import shutil
import os
from pathlib import Path
import click
import pickle
import configparser
import plotly.express as px
import plotly.graph_objects as go
from pybalmorel import MainResults
from pybalmorel.utils import symbol_to_df
from pybalmorel.formatting import balmorel_colours
from Functions.Formatting import newplot, set_style, stacked_bar
from Functions.GeneralHelperFunctions import filter_low_max, AntaresOutput
from Functions.boxplot import get_difference_table
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

balmorel_colours["Spilled"] = "black"
balmorel_colours["WOOD"] = "orange"
balmorel_colours["DUMMY"] = "orange"
balmorel_colours["WOODWASTE"] = "orange"
balmorel_colours["RETORTGAS"] = "orange"
balmorel_colours["CHP-EXTRACTION-CCS"] = "gray"
balmorel_colours["CONDENSING-CCS"] = "gray"
balmorel_colours["WIND OFFSHORE"] = balmorel_colours["WIND-OFF"]
balmorel_colours["WIND ONSHORE"] = balmorel_colours["WIND-ON"]
balmorel_colours["SOLAR PV"] = balmorel_colours["SOLAR-PV"]
balmorel_colours["SPILLED"] = "lightblue"

### ------------------------------- ###
###          1. Functions           ###
### ------------------------------- ###

@click.pass_context
def get_balmorel_results(ctx,
                         obj: pd.DataFrame,
                         cap: pd.DataFrame,
                         cap_F: pd.DataFrame,
                         eltrans: pd.DataFrame,
                         h2trans: pd.DataFrame,
                         dem: pd.DataFrame,
                         pro: pd.DataFrame,
                         proH2: pd.DataFrame,
                         emi: pd.DataFrame,
                         ):
    
    ### 1.2 Load Balmorel Results
    print('Reading results from Balmorel/%s/model/MainResults_%s_Iter%d.gdx..'%(ctx.obj['SC_folder'], ctx.obj['SC'], ctx.obj['i']))
    ws = gams.GamsWorkspace(system_directory=ctx.obj['gams_system_directory'])  
    db = ws.add_database_from_gdx(ctx.obj['wk_dir'] + "/Balmorel/%s/model/MainResults_%s_Iter%d.gdx"%(ctx.obj['SC_folder'], ctx.obj['SC'], ctx.obj['i']))

    ## Get objective function
    temp = symbol_to_df(db, 'OBJ_YCR', ['Y', 'C', 'R', 'Var', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = ctx.obj['i']
    temp = temp.groupby(['Y', 'Var', 'Iter']).aggregate({'Value' : 'sum'})
    obj = pd.concat((obj, temp))

    ## Get Generation & Storage Capacities
    temp = symbol_to_df(db, 'G_CAP_YCRAF', ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity', 'Tech', 'Var', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = ctx.obj['i']
    temp_F = temp[(temp.Tech != 'H2-STORAGE') & (temp.Tech != 'INTRASEASONAL-ELECT-STORAGE') & (temp.Tech != 'INTRASEASONAL-HEAT-STORAGE')]
    temp_F = temp_F.groupby(['Y', 'F', 'Iter']).aggregate({'Value' : 'sum'})
    temp = temp.groupby(['Y', 'Tech', 'Iter']).aggregate({'Value' : 'sum'})
    cap = pd.concat((cap, temp))
    cap_F = pd.concat((cap_F, temp_F))
    
    ## Get Electricity Transmission Capacities
    temp = symbol_to_df(db, 'X_CAP_YCR', ['Y', 'C', 'From', 'To', 'Var', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = ctx.obj['i']
    temp = temp.groupby(['Y', 'To', 'Iter', 'From']).aggregate({'Value' : 'sum'})
    eltrans = pd.concat((eltrans, temp))

    ## Get H2 Transmission Capacities
    try:
        temp = symbol_to_df(db, 'XH2_CAP_YCR', ['Y', 'C', 'From', 'To', 'Var', 'Unit', 'Value'])
        temp.loc[:, 'Iter'] = ctx.obj['i']
        temp = temp.groupby(['Y', 'To', 'Iter', 'From']).aggregate({'Value' : 'sum'})
        h2trans = pd.concat((h2trans, temp))
    except (gams.control.GamsException, ValueError):
        print('No hydrogen transmission')

    ## Get Demand
    temp = symbol_to_df(db, 'EL_DEMAND_YCR', ['Y', 'C', 'R', 'Var', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = ctx.obj['i']
    temp = temp.groupby(['Y', 'Var', 'Iter']).aggregate({'Value' : 'sum'})
    dem = pd.concat((dem, temp))

    ## Get Production
    temp = symbol_to_df(db, 'PRO_YCRAGF', ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity', 'Tech', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = ctx.obj['i']
    temp.loc[:, 'Model'] = 'Balmorel'
    curt = symbol_to_df(db, 'CURT_YCRAGF', ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity', 'Tech', 'Unit', 'Value'])
    curt.loc[:, 'Iter'] = ctx.obj['i']
    curt.loc[:, 'Model'] = 'Balmorel'
    curt = curt.pivot_table(index=['Y', 'Model', 'R', 'F', 'Tech', 'Iter'],
                            values='Value',
                            aggfunc='sum')
    
    # Filter away hydrogen and electrolyser 
    temp2 = temp[(temp.Commodity == 'ELECTRICITY')].copy()
    temp = temp[(temp.Commodity == 'HYDROGEN')]
    temp2 = temp2.groupby(['Y', 'Model', 'R', 'F', 'Tech', 'Iter']).aggregate({'Value' : 'sum'})
    temp = temp.groupby(['Y', 'Model', 'R', 'F', 'Tech', 'Iter']).aggregate({'Value' : 'sum'})
    pro = pd.concat((pro, temp2))
    pro = pd.concat((pro, curt))
    curt = curt.reset_index()
    curt['Tech'] = 'Spilled'
    curt['F'] = 'Spilled'
    curt = curt.pivot_table(index=['Y', 'Model', 'R', 'F', 'Tech', 'Iter'],
                            values='Value',
                            aggfunc='sum')
    curt.loc[:, 'Value'] = -curt.loc[:, 'Value']
    pro = pd.concat((pro, curt))
    proH2 = pd.concat((proH2, temp))

    ## Get Emissions
    temp = symbol_to_df(db, 'EMI_YCRAG', ['Y', 'C', 'R', 'A', 'G', 'F', 'Tech', 'Unit', 'Value'])
    if len(temp) == 0:
        temp = pd.DataFrame(columns=['Y', 'C', 'R', 'A', 'G', 'F', 'Tech', 'Unit', 'Value'],
                            index=[0])
    temp.loc[:, 'Iter'] = ctx.obj['i']
    temp.loc[:, 'Model'] = 'Balmorel'
    temp = temp.groupby(['Model', 'Iter', 'Y', 'R']).aggregate({'Value' : 'sum'})
    emi = pd.concat((emi, temp))
    
    return obj, cap, cap_F, eltrans, h2trans, dem, curt, pro, proH2, emi

@click.pass_context
def get_antares_results(ctx,
                        years: pd.DataFrame,
                        Antobj: pd.DataFrame,
                        pro: pd.DataFrame,
                        emi: pd.DataFrame,):
    
    iteration = ctx.obj['i']
    pro_hourly = {iteration : {}}

    ### 1.3 Load Antares Results
    for year in years:
        
        pro_hourly[iteration][year] = {}
        if not(year == str(ctx.obj['ref_year']) and ctx.obj['i'] != 0):
            ant_output = ctx.obj['antares_output'][ctx.obj['antares_output'].str.find(('eco-' + ctx.obj['SC'] + '_iter%d_y-%s'%(ctx.obj['i'], year)).lower().replace('+', ' ')) != -1].values[0]
            print('\nReading results from %s..\n'%ant_output)
            
            # Load class
            ant_res = AntaresOutput(ant_output)
            
            # Load Antares Costs
            try:
                ant_cost = pd.read_table(os.path.join('Antares/output', ant_output, 'annualSystemCost.txt'),
                        sep=' : ', header=None, engine='python')
                ant_cost['SC'] = ctx.obj['SC']
                ant_cost['Year'] = year
                ant_cost['Iter'] = ctx.obj['i'] 
                Antobj = pd.concat((Antobj, ant_cost), ignore_index=True) 
            except FileNotFoundError:
                # Just a safeguard if i made an error
                print(f'Couldnt store Antares cost output for {ant_output}')
                
            ## Electricity
            for area in ctx.obj['A2B_regi'].keys(): 
                print(f'\nProduction in {area}...\n')
                pro_hourly[iteration][year][area] = {}
                pro_hourly[iteration][year][area]['INTRASEASONAL-ELECT-STORAGE'] = np.zeros(8736)
                try:
                    f = ant_res.load_area_results(area, 'details', 'hourly', ctx.obj['mc_choice']).iloc[:, 5:]
                    
                    ## Thermal Generation
                    for col in [column for column in f.columns if not('.1' in column or '.2' in column or '.3' in column)]:
                        
                        tech = col.split('_')[0].upper()
                        fuel = col.split('_')[1].upper()
                        
                        # Save annual production
                        if not(tech == 'Z'):
                            pro.loc[year, 'Antares', area, fuel, tech, ctx.obj['i']] = f[col].sum()/1e6
                            pro_hourly[iteration][year][area][tech] = f[col].values
                        elif fuel == 'BAT':
                            pro.loc[year, 'Antares', area, 'ELECTRIC', 'BATTERY', ctx.obj['i']] = f[col].sum()/1e6
                            pro_hourly[iteration][year][area]['INTRASEASONAL-ELECT-STORAGE'] += f[col].values
                        elif fuel == 'PSP':
                            pro.loc[year, 'Antares', area, 'ELECTRIC', 'PSP', ctx.obj['i']] = f[col].sum()/1e6
                            pro_hourly[iteration][year][area]['INTRASEASONAL-ELECT-STORAGE'] += f[col].values
                        print(f'Production of {tech} {fuel} was ', f[col].sum()/1e6)
                        
                except FileNotFoundError:
                    # print('No thermal generation in area %s'%area)
                    pass

                f = ant_res.load_area_results(area, 'values', 'hourly', ctx.obj['mc_choice'])
                
                ## CO2
                emi.loc['Antares', ctx.obj['i'], year, area] = f['CO2 EMIS.'].sum() / 1e3 # kton
                    
                ## VRE Generation
                translation = {'WIND ONSHORE' : 'WIND',
                               'WIND OFFSHORE' : 'WIND',
                               'SOLAR PV' : 'SUN'}
                for ren in ['WIND OFFSHORE', 'WIND ONSHORE', 'SOLAR PV']:
                    pro.loc[year, 'Antares', area, translation[ren], ren, ctx.obj['i']] = f[ren].sum()/ 1e6
                    pro_hourly[iteration][year][area][ren] = f[ren].values
                    print(f'Production of {ren} was ', pro.loc[(year, 'Antares', area, translation[ren], ren, ctx.obj['i']), 'Value'].sum())

                ## Spilled Energy (Mainly curtailment of VRE, but in principle thermal must-runs as well)
                spilled = f['SPIL. ENRG'].sum()             
                pro.loc[year, 'Antares', area, 'Spilled', 'Spilled', ctx.obj['i']] = -spilled / 1e6
                pro_hourly[iteration][year][area]['SPILLED'] = f['SPIL. ENRG'].values
                
                ## Hydro
                # In area itself
                pro.loc[year, 'Antares', area, 'WATER', 'HYDRO-RESERVOIRS', ctx.obj['i']] = f.loc[:, 'H. STOR'].sum() / 1e6
                pro_hourly[iteration][year][area]['HYDRO-RESERVOIRS'] = f['H. STOR'].values
                print('Production of hydro-reservoirs was ', pro.loc[(year, 'Antares', area, 'WATER', 'HYDRO-RESERVOIRS', ctx.obj['i']), 'Value'].sum())
                pro.loc[year, 'Antares', area, 'WATER', 'HYDRO-RUN-OF-RIVER', ctx.obj['i']] = f.loc[:, 'H. ROR'].sum() / 1e6
                pro_hourly[iteration][year][area]['HYDRO-RUN-OF-RIVER'] = f['H. ROR'].values
                print('Production of hydro-run-of-river was ', pro.loc[(year, 'Antares', area, 'WATER', 'HYDRO-RUN-OF-RIVER', ctx.obj['i']), 'Value'].sum())
                
    return Antobj, pro, emi, pro_hourly

@click.pass_context
def old_plotting(ctx, obj, cap, cap_F, pro, proH2, eltrans, dem, emi):
    obj.reset_index(inplace=True)
    cap.reset_index(inplace=True)
    cap_F.reset_index(inplace=True)
    pro.reset_index(inplace=True)
    proH2.reset_index(inplace=True)
    eltrans.reset_index(inplace=True)
                                
    ### 1.4 System Costs
    # Filter iterations or not
    idx = filter_low_max(obj, 'Iter', ctx.obj['plot_all'])
    fig = px.bar(obj[idx], x='Y', y='Value', color='Var', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - System Costs (M€)'%ctx.obj['SC'])
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_SystemCosts.html'%ctx.obj['SC'])

    ### 1.5 Generation Capacities wrt. Technology
    # Filter iterations or not
    idx = filter_low_max(cap, 'Iter', ctx.obj['plot_all'])
    idx = idx & (cap.Tech != 'H2-STORAGE') &\
        (cap.Tech != 'INTERSEASONAL-HEAT-STORAGE') &\
        (cap.Tech != 'INTRASEASONAL-HEAT-STORAGE') &\
        (cap.Tech != 'INTRASEASONAL-ELECT-STORAGE') &\
        (cap.Value > 1e-6)
    fig = px.bar(cap[idx], x='Y', y='Value', color='Tech', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - Generation Capacity wrt. Tech (GW)'%ctx.obj['SC'])
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_GenerationTechCapacities.html'%ctx.obj['SC'])

    ### 1.6 Generation Capacities wrt. Fuel
    # Filter iterations or not
    idx = filter_low_max(cap_F, 'Iter', ctx.obj['plot_all'])
    idx = idx & (cap_F.Value > 1e-6)
    fig = px.bar(cap_F[idx], x='Y', y='Value', color='F', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - Generation Capacity wrt. Fuel (GW)'%ctx.obj['SC'])
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_GenerationFuelCapacities.html'%ctx.obj['SC'])

    ### 1.7 Generation wrt. Fuel
    # Electricity
    for model in ['Balmorel', 'Antares']:
        temp = pro[pro.Model == model].groupby(['Y', 'F', 'Iter']).aggregate({'Value' : 'sum'})
        temp.reset_index(inplace=True)
        # temp = temp[~(temp.F == 'Spilled')]
        idx = filter_low_max(temp, 'Iter', ctx.obj['plot_all'])
        idx = idx
        fig = px.bar(temp[idx], x='Y', y='Value', color='F', barmode='stack', facet_col='Iter')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - Generation wrt. Fuel in %s (TWh)'%(ctx.obj['SC'], model))
        fig.show()
        fig.write_html('Workflow/OverallResults/%s_%sGenerationFuel.html'%(ctx.obj['SC'], model))

    # Hydrogen
    for model in ['Balmorel', 'Antares']:
        temp = proH2[proH2.Model == model].groupby(['Y', 'F', 'Iter']).aggregate({'Value' : 'sum'})
        temp.reset_index(inplace=True)
        # temp = temp[~(temp.F == 'Spilled')]
        idx = filter_low_max(temp, 'Iter', ctx.obj['plot_all'])
        idx = idx
        fig = px.bar(temp[idx], x='Y', y='Value', color='F', barmode='stack', facet_col='Iter')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - H2 Generation wrt. Fuel in %s (TWh)'%(ctx.obj['SC'], model))
        fig.show()
        fig.write_html('Workflow/OverallResults/%s_%sH2GenerationFuel.html'%(ctx.obj['SC'], model))

    ### 1.8 Storage Capacities
    # Filter iterations or not
    idx = filter_low_max(cap, 'Iter', ctx.obj['plot_all'])
    idx = idx & (cap.Value > 1e-6) & ((cap.Tech == 'H2-STORAGE') |\
        (cap.Tech == 'INTERSEASONAL-HEAT-STORAGE') |\
        (cap.Tech == 'INTRASEASONAL-HEAT-STORAGE') |\
        (cap.Tech == 'INTRASEASONAL-ELECT-STORAGE'))
    fig = px.bar(cap[idx], x='Y', y='Value', color='Tech', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - Storage Capacity (GWh)'%ctx.obj['SC'])
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_StorageCapacities.html'%ctx.obj['SC'])

    ### 1.9 Electricity Transmission Capacities
    # Filter iterations or not
    temp = eltrans.groupby(['Y', 'Iter', 'To']).aggregate({'Value' : 'sum'}) # Account for double counting
    # temp = eltrans.groupby(['Y', 'From', 'Iter', 'To']).aggregate({'Value' : lambda x: sum(x)/2}) # Account for double counting
    temp.reset_index(inplace=True)
    idx = filter_low_max(temp, 'Iter', ctx.obj['plot_all'])
    idx = idx & (temp.Value > 1e-6)
    fig = px.bar(temp[idx], x='Y', y='Value', color='To', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - 2x El. Transmission Capacity (GW)'%ctx.obj['SC'])
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_ElTransCapacities.html'%ctx.obj['SC'])
    
    # ### 1.10 Hydrogen Transmission Capacities
    # try:
    #     # Filter iterations or not
    #     temp = h2trans.groupby(['Y', 'Iter', 'To']).aggregate({'Value' : 'sum'}) # Account for double counting
    #     # temp = eltrans.groupby(['Y', 'From', 'Iter', 'To']).aggregate({'Value' : lambda x: sum(x)/2}) # Account for double counting
    #     temp.reset_index(inplace=True)
    #     idx = filter_low_max(temp, 'Iter', ctx.obj['plot_all'])
    #     idx = idx & (temp.Value > 1e-6)
    #     fig = px.bar(temp[idx], x='Y', y='Value', color='To', barmode='stack', facet_col='Iter')
    #     fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - 2x H2 Transmission Capacity (GW)'%ctx.obj['SC'])
    #     fig.show()
    #     fig.write_html('Workflow/OverallResults/%s_H2TransCapacities.html'%ctx.obj['SC'])
    # except:
    #     pass
    
    ### 1.11 Electricity Demand
    # Filter iterations or not
    temp = dem
    temp.reset_index(inplace=True)
    idx = filter_low_max(temp, 'Iter', ctx.obj['plot_all'])
    idx = idx & (temp.Value > 1e-6)
    fig = px.bar(temp[idx], x='Y', y='Value', color='Var', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - Electricity Demand (GWh)'%ctx.obj['SC'])
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_ElecDemand.html'%ctx.obj['SC'])
    
    
    ### 1.12 Emissiongs
    # Filter iterations or not
    temp = emi.reset_index()
    for model in temp.Model.unique():
        temp2 = temp[temp.Model == model]
        # temp.reset_index(inplace=True)
        idx = filter_low_max(temp2, 'Iter', ctx.obj['plot_all'])
        idx = idx & (temp2.Value > 1e-6)
        fig = px.bar(temp2[idx], x='Y', y='Value', color='R', barmode='stack', facet_col='Iter')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=ctx.obj['plotly_theme'],title='%s - %s Emissions (ktonCO2)'%(ctx.obj['SC'], model))
        fig.show()
        fig.write_html('Workflow/OverallResults/%s_%sEmissions.html'%(ctx.obj['SC'], model))
    

    ### 1.12 Unserved Energy
    # for carrier in ['Elec', 'H2']:
    #     f = pd.read_csv('OverallResults/%s_%sNotServedMWh.csv'%(ctx.obj['SC'], carrier))
    #     f.columns = ['Iter'] + list(f.columns[1:])
    #     f.iloc[:, 1:] = f.iloc[:, 1:] / 1e3 # GWh
    #     f['Iter'] = f['Iter'].astype(int)

    #     fig, ax = newplot(figsize=figsize, fc=ctx.obj['fc'])
    #     f.plot(x=['Y', 'Iter'], ax=ax, stacked=True, kind='bar', zorder=5)
    #     ax.legend(loc='center', bbox_to_anchor=(.5, 1.25), ncol=3)
    #     ax.set_ylabel('Unsupplied %s (GWh)'%carrier)
    #     ax.set_xlabel('Iteration')
    #     ax.set_title(year)
    #     # ax.set_xticks(xticks)
    #     fig.savefig('OverallResults/%s_Unserved%s.png'%(ctx.obj['SC'], carrier), 
    #                 bbox_inches='tight', transparent=True)


    ### ------------------------------- ###
    ###         2. Pareto Front         ###
    ### ------------------------------- ###

    ## Read LOLD
    fLOLD = pd.read_csv('Workflow/OverallResults/%s_LOLD.csv'%ctx.obj['SC'], index_col=0)


    ### 2.1 Save pareto front data
    # PF = pd.DataFrame({})
    # fig, ax = newplot(figsize=figsize, fc=ctx.obj['fc'])
    # for year in Y:
    #     temp = fLOLD[(fLOLD.Year == int(year)) & (fLOLD.Carrier == 'Electricity')].groupby(['Iter', 'Year']).aggregate({'Value (h)' : 'sum'})
        
    #     PF = pd.concat((PF, pd.DataFrame({'Iter' : np.arange(len(temp)),
    #                     'SC' : [ctx.obj['SC']]*len(temp),
    #                     'Year' : [year]*len(temp),
    #                     'ElecLOLD_h' : temp.values[:,0],
    #                     'SystemCost_MEUR' : obj[obj.Y == year].groupby(by=['Iter', 'Y']).aggregate({'Value' : 'sum'}).values[:,0]})))

    #     ax.plot(PF.ElecLOLD_h,
    #             PF.SystemCost_MEUR,
    #             'o')
    #     ax.set_ylabel('System Costs (MEUR)')
    #     ax.set_xlabel('Loss of Load Duration (h)')
    #     ax.set_xscale('log')
    #     fig.savefig('Workflow/OverallResults/%s_ParetoFront.png'%ctx.obj['SC'], 
    #                 bbox_inches='tight', transparent=True)
    # PF.to_csv('Workflow/OverallResults/%s_ParetoFront.csv'%ctx.obj['SC'], index=False)


    ### 2.2 Comparison between other pareto fronts
    if ctx.obj['plotPFcomparison']:
        colours = [(0.5, .85, 0.5), (.85, 0.5, 0.5), (0.5, 0.5, .85), (.85, .5, .85)]
        
        for year in ctx.obj['years']:
            # PF = pd.DataFrame()
            pfdata = pd.Series(os.listdir('Workflow/OverallResults'))[pd.Series(os.listdir('Workflow/OverallResults')).str.find('ParetoFront.csv') != -1]
            fig, ax = newplot(figsize=(7,3), fc=ctx.obj['fc'])
            
            j = 0
            for pf in pfdata: 
                if (pf != 'FictDemMarketValue_ParetoFront.csv'):
                    pf_name = pf.replace('_ParetoFront.csv', '')    
                    # PF = pd.concat((PF, pd.read_csv('Workflow/OverallResults/%s'%pf)))
                    PF = pd.read_csv('Workflow/OverallResults/%s'%pf)
                    PF = PF[PF.Year == int(year)]
                    ax.plot(PF.ElecLOLD_h, PF.SystemCost_MEUR, 'o', label=pf.replace('_ParetoFront.csv', ''),
                            markersize=2, color=colours[j])
                    j += 1
                
            ax.set_title(year)
            ax.legend()
            # ax.legend(('Capacity Credit', 'Fictive Demand + Market Value', 'Fictive Demand'))
            ax.set_ylabel('System Cost (MEUR)')
            ax.set_xlabel('Loss of Load Duration (h)')
            ax.set_xscale('log')
            fig.savefig('Workflow/OverallResults/PFComparison_%s.png'%year, transparent=True,
                        bbox_inches='tight')

       
@click.pass_context
def store_and_zip(ctx):
    ### ------------------------------- ###
    ###        4. Collect Results       ###
    ### ------------------------------- ###

    ### 4.1 Collect LOLD .csv's
    l = pd.Series(os.listdir('Workflow/OverallResults'))
    lElec = l[l.str.find('_ElecLOLD.csv') != -1]

    # Elec
    df = pd.DataFrame()
    for file in lElec:
        temp = pd.read_csv('Workflow/OverallResults/' + file)
        temp.columns = ['SC', 'Iter', 'Year', 'Region', 'Value']
        temp['SC'] = file.split('_ElecLOLD')[0]
        df = df._append(temp, ignore_index=True)
    df.to_csv('Workflow/OverallResults/ElecLOLD_AllSC.csv', index=False)

    # H2
    df = pd.DataFrame()
    lH2 = l[l.str.find('_H2LOLD.csv') != -1]
    for file in lH2:
        temp = pd.read_csv('Workflow/OverallResults/' + file)
        temp.columns = ['SC', 'Iter', 'Year', 'Region', 'Value']
        temp['SC'] = file.split('_H2LOLD')[0]
        df = df._append(temp, ignore_index=True)
    df.to_csv('Workflow/OverallResults/H2LOLD_AllSC.csv', index=False)


    ### 4.2 Collect Antares System Costs
    l = pd.Series(os.listdir('Antares/output'))

    df = pd.DataFrame()
    for file in l:
        if '_iter' in file and '_y-' in file:    
            try:
                temp = pd.read_table('Antares/output/%s/annualSystemCost.txt'%file, header=None)
                temp = float(temp.loc[0,0].lstrip('EXP : '))
                
                SCENARIO = file.split('eco-')[1]
                year = int(SCENARIO.split('_y-')[1])
                SCENARIO = SCENARIO.split('_y-')[0]
                iter = int(SCENARIO.split('_iter')[1])
                SCENARIO = SCENARIO.split('_iter')[0]
                
                df = df._append(pd.DataFrame({'SC' : SCENARIO, 'Y' : year, 'Iter' : iter, 'ObjCost' : temp},
                                            index=[0]), ignore_index=True)
            except FileNotFoundError:
                pass

    df.to_csv('Workflow/OverallResults/AntaresSystemCost.csv', index=False)


    ### 4.3 Zip everything (linux commands)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M")
    zip_filename = 'Workflow/OverallResults/' + dt_string + '_%s_Results.zip'%ctx.obj['SC']
    errors = False
    results = ['AntaresEmissions.html',
                'BalmorelEmissions.html',
                'ElecDemand.html',
                'H2TransCapacities.html',
                'ElTransCapacities.html',
                'AntaresGenerationFuel.html',
                'AntaresH2GenerationFuel.html',
                'StorageCapacities.html',
                'BalmorelGenerationFuel.html',
                'BalmorelH2GenerationFuel.html',
                'GenerationFuelCapacities.html',
                'GenerationTechCapacities.html',
                'SystemCosts.html',
                'results.pkl',
                'ProcessTime.csv',
                'ElecNotServedMWh.csv',
                'H2NotServedMWh.csv',
                'MV.csv',
                'LOLD.csv']

    if ctx.obj['USE_CAPCRED']:
        results.append('CC.pkl')
        results.append('ResMar.csv')
        if ctx.obj['USE_H2CAPCRED']:
            results.append('CCH2.pkl')


    if ctx.obj['zip_files']:
        
        # Zip Overall Results
        print('Zipping overall results..')
        for result in results:
            
            out = os.system('zip -r -q "%s" "Workflow/OverallResults/%s_%s"'%(zip_filename, ctx.obj['SC'], result)) 
            
            if out != 0:
                errors = True
                break
            
            if ctx.obj['del_files']:
                print('\nDeleting..')
                os.remove(os.path.join(ctx.obj['wk_dir'], 'Workflow/OverallResults', ctx.obj['SC'] + '_' + result))
                
        # Zip configfile
        out = os.system('zip -r -q "%s" "Workflow/MetaResults/%s_meta.ini"'%(zip_filename, ctx.obj['SC'])) 
        if ctx.obj['del_files']:
            os.remove(os.path.join(ctx.obj['wk_dir'], 'Workflow/MetaResults', ctx.obj['SC'] + '_meta.ini'))
        
        # Zip Balmorel Results
        if not(errors):
            for j in ctx.obj['iter']:
                balm_res = "Balmorel/%s/model/MainResults_%s_Iter%d.gdx"%(ctx.obj['SC_folder'], ctx.obj['SC'], j)
                print('Zipping %s..'%balm_res)
                
                out = os.system('zip -r -q "%s" "%s"'%(zip_filename, balm_res)) 

                if out != 0:
                    errors = True
                    break
                
                if ctx.obj['del_files']:
                    print('\nDeleting..')
                    os.remove(os.path.join(ctx.obj['wk_dir'], balm_res))
                
        # Zip Antares Results
        if not(errors):
            for ant_file in ctx.obj['antares_output']:
                print('Zipping %s..'%ant_file)
                out = os.system('zip -r -q "%s" "Antares/output/%s"'%(zip_filename, ant_file)) 
        
                if out != 0:
                    errors = True
                    break
                
                if ctx.obj['del_files']:
                    print('\nDeleting..')
                    shutil.rmtree(os.path.join(ctx.obj['wk_dir'], 'Antares/output', ant_file))

def plot_annual_electricity_generation(results: dict, **kwargs):
    pro = results["pro"]
    fig, ax = plt.subplots()
    pro.pivot_table(index="Model", columns="F", values="Value", aggfunc="sum").plot(
        ax=ax, kind="bar", stacked=True, color=balmorel_colours
    )
    print(pro.pivot_table(index=["Model", "F"], values="Value", aggfunc="sum"))
    ax.set_facecolor(kwargs.get("facecolor", "none"))
    fig.set_facecolor(kwargs.get("facecolor", "none"))
    ax.set_ylabel("Electricity Generation (TWh)")
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")

    return fig, ax


def plot_antares_hourly_electricity_generation(
    production_hourly: dict,
    iteration: int = 0,
    week: int = 1,
    year: str = "2050",
    regions: str | list = "all",
    **kwargs,
):
    antares_production = production_hourly[iteration][year]
    df = None

    # Aggregate to regional choice
    if type(regions) is str:
        if regions.lower() != "all":
            df = pd.DataFrame(antares_production[regions])
        else:
            regions = list(antares_production.keys())

    if type(regions) is list:
        df = pd.DataFrame(antares_production[regions[0]])
        for region in regions[1:]:
            df = df.add(pd.DataFrame(antares_production[region]), fill_value=0)

    if df is None:
        raise ValueError("Wrong choice of regions")

    # Exclude SPILLED and make into GW
    df = (
        df
        .drop(columns='SPILLED')
        .div(1e3)
    )

    # Plot timeseries
    with open("test.txt", "w") as f:
        f.write(df.to_string())

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (9, 3)))
    df.loc[(week - 1) * 168 : week * 168].plot(
        ax=ax, stacked=True, kind="area", color=balmorel_colours
    )
    ax.set_facecolor(kwargs.get("facecolor", "none"))
    ax.set_ylabel("Power (GW)")
    ax.set_xlim((week - 1) * 168, week * 168)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncols=4)
    fig.set_facecolor(kwargs.get("facecolor", "none"))

    return fig, ax


def plot_balmorel_hourly_electricity_generation(
    scenario: str, iteration: int = 0, week: int = 1, year: str = "2050", region='ALL', gams_system_directory: str = '/appl/gams/47.6.0'
):
    filename = f"MainResults_{scenario}_Iter{iteration}.gdx"
    paths = [path for path in Path('Balmorel').glob(f'**/model/{filename}')]

    if len(paths) > 1:
        raise ValueError(f"Found multiple {filename}!\n{paths}")

    result = MainResults(filename, str(paths[0].parent), scenario,
                        system_directory=gams_system_directory)

    fig, ax = result.plot_profile("electricity", int(year), scenario, region=region)

    ax.set_xlim((week - 1) * 168, week * 168)

    return fig, ax

### ------------------------------- ###
###           2. Main CLI           ###
### ------------------------------- ###


@click.group()
@click.pass_context
@click.option("--dark", is_flag=True, default=False, help="Dark plots?")
def CLI(ctx, dark):
    # Context manager
    ctx.ensure_object(dict)

    if dark:
        ctx.obj["fc"], ctx.obj["plotly_theme"] = set_style("ppt")
    else:
        ctx.obj["fc"], ctx.obj["plotly_theme"] = set_style("report")

    ## 1.0 Plot design
    ctx.obj["figsize"] = (10, 5)


@CLI.command()
@click.argument('scenario', type=str)
@click.pass_context
def collect_results(ctx, scenario: str):
    
    Config = configparser.ConfigParser()
    Config.read('Workflow/MetaResults/%s_meta.ini'%scenario)
    ctx.obj['SC_folder'] = Config.get('RunMetaData', 'SC_Folder')
    ctx.obj['USE_CAPCRED']   = Config.getboolean('PostProcessing', 'Capacitycredit')
    ctx.obj['USE_H2CAPCRED']   = Config.getboolean('PostProcessing', 'H2Capacitycredit')

    # Analysis Settings
    ctx.obj['plotprofiles'] = 'n' # Choose whether to plot profiles or not
    ctx.obj['plotantaresViz'] = 'n'
    ctx.obj['plotPFcomparison'] = False
    ctx.obj['plot_all'] = Config.getboolean('Analysis', 'plot_all')
    ctx.obj['zip_files'] = Config.getboolean('Analysis', 'zip_files')
    ctx.obj['del_files'] = Config.getboolean('Analysis', 'del_files')
        
    # Years
    years = np.array(Config.get('RunMetaData', 'Y').split(',')).astype(int)
    years.sort()
    years = years.astype(str)
    ctx.obj['years'] = years
    ctx.obj['ref_year'] = Config.getint('RunMetaData', 'ref_year')
    ctx.obj['gams_system_directory'] = Config.get('RunMetaData', 'gams_system_directory')

    ### 0.1 Working Directory
    ctx.obj['wk_dir'] = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


    ### 0.2 Antares region mapping
    with open(ctx.obj['wk_dir'] + '/Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
        ctx.obj['A2B_regi'] = pickle.load(f)
    with open(ctx.obj['wk_dir'] + '/Pre-Processing/Output/A2B_regi_h2.pkl', 'rb') as f:
        ctx.obj['A2B_regi_h2'] = pickle.load(f)

    # Full antares region list
    with open(ctx.obj['wk_dir'] + '/Pre-Processing/Output/antreglist.pkl', 'rb') as f:
        ctx.obj['ANTREGLIST'] = pickle.load(f)

    ### 0.4 Which results to import?
    ant_out = pd.Series(os.listdir(ctx.obj['wk_dir'] + '/Antares/output'))
    ant_out = ant_out[ant_out.str.find(('eco-' + scenario + '_iter').lower().replace('+',' ')) != -1].sort_values(ascending=False)
    ctx.obj['antares_output'] = ant_out

    # Find iterations
    iters = list(ant_out.str.split('_iter', expand=True).iloc[:,1].str.split('_y-',expand=True).iloc[:,0].astype(int)) 
    iters = pd.Series(iters).unique()
    iters.sort()
    ctx.obj['iters'] = iters
    print('\nIterations as read from Antares output: %d'%len(iters))

    # Save to context
    ctx.obj['SC'] = scenario
    
    # 1. Collect Annual Values
    ## 1.1 Placeholders and useful data
    obj = pd.DataFrame({})
    Antobj = pd.DataFrame({})
    cap = pd.DataFrame({})
    cap_F = pd.DataFrame({})
    eltrans = pd.DataFrame({})
    h2trans = pd.DataFrame({})
    dem = pd.DataFrame({})
    pro = pd.DataFrame({})
    proH2 = pd.DataFrame({})
    emi = pd.DataFrame({})

    # uniq_fuels = np.array(['biogas', 'biooil', 'coal', 'electric', 'fueloil', 'heat',
    #        'hydrogen', 'lightoil', 'lignite', 'muniwaste', 'natgas',
    #        'nuclear', 'straw', 'sun', 'wasteheat', 'water', 'wind',
    #        'woodchips', 'woodpellets', 'woodwaste'], dtype=object)
    ctx.obj['mc_choice'] = 'mc-all' # MC year in Antares for generation results
    for j in iters:
        ctx.obj['i'] = j
        obj, cap, cap_F, eltrans, h2trans, dem, curt, pro, proH2, emi = get_balmorel_results(obj, cap, cap_F, eltrans, h2trans, dem, pro, proH2, emi)
        
        Antobj, pro, emi, pro_hourly = get_antares_results(years, 
                                                           Antobj, 
                                                           pro, 
                                                           emi)
        
    # Store pickle file with all dataframes
    with open('Workflow/OverallResults/%s_results.pkl'%scenario, 'wb') as f:
        pickle.dump({'obj' : obj,
                    'Aobj' : Antobj,
                    'capT' : cap,
                    'capF' : cap_F,
                    'eltrans' : eltrans,
                    'h2trans' : h2trans,
                    'dem' : dem,
                    'pro' : pro,
                    'proh2' : proH2,
                    'pro_hourly' : pro_hourly,
                    'emi' : emi}, f)

@CLI.command()
@click.argument("scenario", type=str, required=True)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Collect results again, even if it exists",
)
@click.pass_context
def plot(ctx, scenario, overwrite):
    # Collect results if overwrite or if it doesn't exist
    result_path = Path(f"Workflow/OverallResults/{scenario}_results.pkl")
    if not result_path.exists() or overwrite:
        print(f"Collecting {scenario} results...")
        ctx.invoke(collect_results, scenario=scenario)

    # Load results
    with open(str(result_path), "rb") as f:
        results = pickle.load(f)

    # Annual electricity generation
    fig, _ = plot_annual_electricity_generation(results, facecolor=ctx.obj["fc"])
    fig.savefig(
        f"Workflow/OverallResults/{scenario}_elec_gen.png",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    region = 'all'
    for week in range(1, 53):
        # Electricity generating profile per week
        fig_ant, ax_ant = plot_antares_hourly_electricity_generation(
            results["pro_hourly"], regions=region,week=week,
        )

        fig_balm, ax_balm = plot_balmorel_hourly_electricity_generation(
            scenario, week=week, region=region
        )

        # Find highest ylim
        ant_ylims = ax_ant.get_ylim()
        balm_ylims = ax_balm.get_ylim()
        max_ylim = np.max([ant_ylims[1], balm_ylims[1]])
        ax_ant.set_ylim(0, max_ylim)
        ax_balm.set_ylim(0, max_ylim)

        fig_ant.savefig(
            f"Workflow/OverallResults/{scenario}_W{week}_{region}_antares_elec_gen_hourly.png",
            bbox_inches="tight",
            transparent=True,
        )
        fig_balm.savefig(
            f"Workflow/OverallResults/{scenario}_W{week}_{region}_balmorel_elec_gen_hourly.png",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig_ant)
        plt.close(fig_balm)

@CLI.command()
def plot_virginie_clustering_table():
    filename = 'Workflow/OverallResults/PtX_demand_comparison_virginie_clustering.csv'
    df_diff, df_diff_mean = get_difference_table(filename, '# Demand Curves', r'cl(\d+)', int)
    print(df_diff_mean.to_string())
    df_diff_mean.loc['noh', [4, 8, 52, 168]] = df_diff_mean.loc['noh_fullyear', [4, 8, 52, 168]].values
    
    # print(df_diff.loc[['noh2', 'h2', 'h2_lss']].round())
    print(df_diff_mean.loc[['noh', 'noh2', 'h2', 'h2_lss', 'h2_lss_h2t']].round().to_string())

@CLI.command()
def plot_virginie_data_table():
    filename = 'Workflow/OverallResults/PtX_demand_comparison_virginie_data.csv'
    df_diff, df_diff_mean = get_difference_table(filename, 'Data', r'fullyear_(.+)\_iter0', str)

    df_diff_mean.loc[:, 'h2surhsur'] = df_diff_mean.loc[:, ['h2surhsur_oldrounding', 'h2surhsur_cl168']].sum(axis=1)
    df_diff_mean.loc[:, 'h2vrehsur'] = df_diff_mean.loc[:, ['h2vrehsur_oldrounding', 'h2vrehsur_cl168']].sum(axis=1)
    df_diff_mean.loc[:, 'h2surhexo'] = df_diff_mean.loc[:, ['h2surhexo_oldrounding', 'h2surhexo_cl168']].sum(axis=1)
    df_diff_mean.loc[:, 'h2vrehexo'] = df_diff_mean.loc[:, ['h2vrehexo_oldrounding', 'h2vrehexo_cl168']].sum(axis=1)
    
    # print(df_diff.loc[['noh', 'noh2', 'h2', 'h2_lss']].round())
    print(df_diff_mean.loc[['noh', 'noh2', 'h2', 'h2_lss', 'h2_lss_h2t'], ['h2vrehexo', 'h2vrehsur', 'h2surhexo', 'h2surhsur']].round().to_string())

@CLI.command()
@click.argument('weather_year', type=int)
def plot_multiweather_table(weather_year: int):
    filename = f'Workflow/OverallResults/PtX_demand_comparison_multiweather_{weather_year}trained.csv'
    df_diff, df_diff_mean = get_difference_table(filename, 
                                                 'Data',
                                                 r'\_dispatch\_WY(.+)\_Iter0',
                                                 int,
                                                 r'eco-(.+)\_wy',
                                                 'BalmorelFile')

    print(f'Trained on weather year {weather_year}')
    print(df_diff_mean.mean().round().to_string())
    print(f'Average deviation for trained weather year: {df_diff_mean.loc[:, weather_year].mean().round(2)}')
    print(f'Average deviation for other weather years:  {df_diff_mean.loc[:, df_diff_mean.columns.drop(weather_year)].mean().mean().round(2)}')

if __name__ == "__main__":
    CLI()
