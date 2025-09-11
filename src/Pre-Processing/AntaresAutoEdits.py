"""
TITLE

Description

Created on 20.08.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
### ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import click
import rpy2.robjects as robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
import pickle as pkl
from pybalmorel import Balmorel
from pybalmorel.utils import symbol_to_df
from configparser import ConfigParser

### ------------------------------- ###
###          1. Functions           ###
### ------------------------------- ###

def load_possible_investments(gams_system_directory: str ='/opt/gams/48.5'):
    
    model = Balmorel('Balmorel', gams_system_directory=gams_system_directory)
    model.load_incfiles('base')
    df = symbol_to_df(model.input_data['base'], 'XINVCOST')
    
    return df

### ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###

@click.group()
@click.option('--dark-style', is_flag=True, required=False, help='Dark plot style')
def main(dark_style: bool):
    # Set global style of plot
    if dark_style:
        plt.style.use('dark_background')
        fc = 'none'
    else:
        fc = 'white'
        
    # Load R packages to R instance
    importr("base")
    importr("antaresEditObject")
    
@main.command()
def regions():

    # Load all possible investments (remember to have a full Balmorel dataset!)
    df = load_possible_investments().query('IRRRE != "ROW" and IRRRI != "ROW" and YYY == "2050"').drop_duplicates()
    regions = set(df.IRRRE.unique()) | set(df.IRRRI.unique()) 
    print(regions)

    
    # Load simulation
    robjects.r('antaresRead::setSimulationPath(path="Antares", simulation="input")')

    for region in regions:
        
        # Create region if it doesn't exist    
        try:
            out = robjects.r(f'createArea(name = "{region}")')
        except RRuntimeError as exception:
            print(f'{region} is probably already created, check error message:\n{exception}')
            
        # Create PtX regions, if they don't exist
        for commodity in ['HEAT', 'HYDROGEN']:
            try:
                out = robjects.r(f'createArea(name = "{region}_{commodity}")')
                
                # Append to list of links
                df = pd.concat((df, pd.DataFrame({'YYY' : ["2050"], 'IRRRE' : [region], 'IRRRI' : [f'{region}_{commodity}'], 'Value' : [1]})), ignore_index=True)
                
            except RRuntimeError as exception:
                print(f'{region}_{commodity} is probably already created, check error message:\n{exception}')
            
        # Add links to battery and PSP storage
        df = pd.concat((df, pd.DataFrame({'YYY' : ["2050"], 'IRRRE' : [region], 'IRRRI' : ['00_BAT_STO'], 'Value' : [1]})), ignore_index=True)
        df = pd.concat((df, pd.DataFrame({'YYY' : ["2050"], 'IRRRE' : [region], 'IRRRI' : ['00_PSP_STO'], 'Value' : [1]})), ignore_index=True)
            
        # Create link if it doesn't exist
        for importing_link in df.query(f'IRRRE == "{region}"')['IRRRI']:
            try:
                out = robjects.r(f'''
                                createLink(
                                    from = "{region}",
                                    to = "{importing_link}",
                                    propertiesLink = propertiesLinkOptions(
                                        hurdles_cost = FALSE,
                                        transmission_capacities = "ignore"
                                    ),
                                    dataLink = NULL
                                )
                                ''')
            except RRuntimeError as exception:
                print(f'{region}-{importing_link} probably exist already. Check error message:\n{exception}')


if __name__ == '__main__':
    
    run_old_code = False
    
    if not(run_old_code):
        main()
        
    else:

        ### ------------------------------- ###
        ###       2. AntaresEditObject      ###
        ### ------------------------------- ###


        importr("base")
        importr("antaresEditObject")

        out = robjects.r('''
                        antaresRead::setSimulationPath(path="Antares", simulation="input")
                        ''')


        # 2.0 Create Areas

        areas = ['flexdem']

        for a in areas:
            try:
                out = robjects.r('''
                                createArea(name = "%s")
                                '''%a)
            except:
                print('%s is already created'%a)

        #%% 2.1 Create or Edit Links
        ## Virtual links for electrolyser modelling


        virtuallinksh2 = {'z_h2_c3_al00' : 'z_taking',
                        'z_h2_c3_at00' : 'z_taking',
                        'z_h2_c3_ba00' : 'z_taking',
                        'z_h2_c3_be00' : 'z_taking',
                        'z_h2_c3_bg00' : 'z_taking',
                        'z_h2_c3_ch00' : 'z_taking',
                        'z_h2_c3_cy00' : 'z_taking',
                        'z_h2_c3_cz00' : 'z_taking',
                        'z_h2_c3_de00' : 'z_taking',
                        'z_h2_c3_dkw1' : 'z_taking',
                        'z_h2_c3_dke1' : 'z_taking',
                        'z_h2_c3_ee00' : 'z_taking',
                        'z_h2_c3_es00' : 'z_taking',
                        'z_h2_c3_fi00' : 'z_taking',
                        'z_h2_c3_fr00' : 'z_taking',
                        'z_h2_c3_gr00' : 'z_taking',
                        'z_h2_c3_hr00' : 'z_taking',
                        'z_h2_c3_hu00' : 'z_taking',
                        'z_h2_c3_ie00' : 'z_taking',
                        'z_h2_c3_itcn' : 'z_taking',
                        'z_h2_c3_itcs' : 'z_taking',
                        'z_h2_c3_itn1' : 'z_taking',
                        'z_h2_c3_its1' : 'z_taking',
                        'z_h2_c3_itsi' : 'z_taking',
                        'z_h2_c3_lt00' : 'z_taking',
                        'z_h2_c3_lu00' : 'z_taking',
                        'z_h2_c3_lv00' : 'z_taking',
                        'z_h2_c3_me00' : 'z_taking',
                        'z_h2_c3_mk00' : 'z_taking',
                        'z_h2_c3_mt00' : 'z_taking',
                        'z_h2_c3_nl00' : 'z_taking',
                        'z_h2_c3_nos0' : 'z_taking',
                        'z_h2_c3_nom1' : 'z_taking',
                        'z_h2_c3_non1' : 'z_taking',
                        'z_h2_c3_pl00' : 'z_taking',
                        'z_h2_c3_pt00' : 'z_taking',
                        'z_h2_c3_ro00' : 'z_taking',
                        'z_h2_c3_rs00' : 'z_taking',
                        'z_h2_c3_se01' : 'z_taking',
                        'z_h2_c3_se02' : 'z_taking',
                        'z_h2_c3_se03' : 'z_taking',
                        'z_h2_c3_se04' : 'z_taking',
                        'z_h2_c3_si00' : 'z_taking',
                        'z_h2_c3_sk00' : 'z_taking',
                        'z_h2_c3_tr00' : 'z_taking',
                        'z_h2_c3_uk00' : 'z_taking',
                        'z_h2_c3_ukni' : 'z_taking'}


        #%%
        for link in virtuallinksh2.keys():
            try:
                out = robjects.r('''
                                createLink(
                                    from = "%s",
                                    to = "%s",
                                    propertiesLink = propertiesLinkOptions(
                                        hurdles_cost = FALSE,
                                        transmission_capacities = "ignore"
                                    ),
                                    dataLink = NULL
                                )
                                '''%(link, virtuallinksh2[link]))
            except:
                # print('Probably exist already. Trying to edit instead.')
                out = robjects.r('''
                                editLink(
                                    from = "%s",
                                    to = "%s",
                                    transmission_capacities = "ignore"
                                    )
                                '''%(link, virtuallinksh2[link]))

        #%%
        virtuallinksh2 = {'z_h2_c3_al00' : 'x_c3',
                        'z_h2_c3_at00' : 'x_c3',
                        'z_h2_c3_ba00' : 'x_c3',
                        'z_h2_c3_be00' : 'x_c3',
                        'z_h2_c3_bg00' : 'x_c3',
                        'z_h2_c3_ch00' : 'x_c3',
                        'z_h2_c3_cy00' : 'x_c3',
                        'z_h2_c3_cz00' : 'x_c3',
                        'z_h2_c3_de00' : 'x_c3',
                        'z_h2_c3_dkw1' : 'x_c3',
                        'z_h2_c3_dke1' : 'x_c3',
                        'z_h2_c3_ee00' : 'x_c3',
                        'z_h2_c3_es00' : 'x_c3',
                        'z_h2_c3_fi00' : 'x_c3',
                        'z_h2_c3_fr00' : 'x_c3',
                        'z_h2_c3_gr00' : 'x_c3',
                        'z_h2_c3_hr00' : 'x_c3',
                        'z_h2_c3_hu00' : 'x_c3',
                        'z_h2_c3_ie00' : 'x_c3',
                        'z_h2_c3_itcn' : 'x_c3',
                        'z_h2_c3_itcs' : 'x_c3',
                        'z_h2_c3_itn1' : 'x_c3',
                        'z_h2_c3_its1' : 'x_c3',
                        'z_h2_c3_itsi' : 'x_c3',
                        'z_h2_c3_lt00' : 'x_c3',
                        'z_h2_c3_lu00' : 'x_c3',
                        'z_h2_c3_lv00' : 'x_c3',
                        'z_h2_c3_me00' : 'x_c3',
                        'z_h2_c3_mk00' : 'x_c3',
                        'z_h2_c3_mt00' : 'x_c3',
                        'z_h2_c3_nl00' : 'x_c3',
                        'z_h2_c3_nos0' : 'x_c3',
                        'z_h2_c3_nom1' : 'x_c3',
                        'z_h2_c3_non1' : 'x_c3',
                        'z_h2_c3_pl00' : 'x_c3',
                        'z_h2_c3_pt00' : 'x_c3',
                        'z_h2_c3_ro00' : 'x_c3',
                        'z_h2_c3_rs00' : 'x_c3',
                        'z_h2_c3_se01' : 'x_c3',
                        'z_h2_c3_se02' : 'x_c3',
                        'z_h2_c3_se03' : 'x_c3',
                        'z_h2_c3_se04' : 'x_c3',
                        'z_h2_c3_si00' : 'x_c3',
                        'z_h2_c3_sk00' : 'x_c3',
                        'z_h2_c3_tr00' : 'x_c3',
                        'z_h2_c3_uk00' : 'x_c3',
                        'z_h2_c3_ukni' : 'x_c3'}

        for link in virtuallinksh2.keys():
            try:
                out = robjects.r('''
                                createLink(
                                    from = "%s",
                                    to = "%s",
                                    propertiesLink = propertiesLinkOptions(
                                        hurdles_cost = FALSE,
                                        transmission_capacities = "ignore"
                                    ),
                                    dataLink = NULL
                                )
                                '''%(virtuallinksh2[link], link))
            except:
                print('Probably exist already. Trying to edit instead.')
                out = robjects.r('''
                                editLink(
                                    from = "%s",
                                    to = "%s",
                                    transmission_capacities = "ignore"
                                    )
                                '''%(link, virtuallinksh2[link]))
        #%%


        A2B_regi = pkl.load(open('Pre-Processing/Output/A2B_regi.pkl', 'rb'))

        virtuallinksel = {key : 'flexdem' for key in A2B_regi.keys()}


        # #%%
        # virtuallinksel = {'al00'.upper() :  'x_c3',
        #                   'at00'.upper() :  'x_c3',
        #                   'ba00'.upper() :  'x_c3',
        #                   'be00'.upper() :  'x_c3',
        #                   'bg00'.upper() :  'x_c3',
        #                   'ch00'.upper() :  'x_c3',
        #                   'cy00'.upper() :  'x_c3',
        #                   'cz00'.upper() :  'x_c3',
        #                   'de00'.upper() :  'x_c3',
        #                   'dkw1'.upper() :  'x_c3',
        #                   'dke1'.upper() :  'x_c3',
        #                   'ee00'.upper() :  'x_c3',
        #                   'es00'.upper() :  'x_c3',
        #                   'fi00'.upper() :  'x_c3',
        #                   'fr00'.upper() :  'x_c3',
        #                   'fr15'.upper() :  'x_c3',
        #                   'gr00'.upper() :  'x_c3',
        #                   'gr03'.upper() :  'x_c3',
        #                   'hr00'.upper() :  'x_c3',
        #                   'hu00'.upper() :  'x_c3',
        #                   'ie00'.upper() :  'x_c3',
        #                   'itca'.upper() :  'x_c3',
        #                   'itcn'.upper() :  'x_c3',
        #                   'itcs'.upper() :  'x_c3',
        #                   'itn1'.upper() :  'x_c3',
        #                   'its1'.upper() :  'x_c3',
        #                   'itsa'.upper() :  'x_c3',
        #                   'itsi'.upper() :  'x_c3',
        #                   'lt00'.upper() :  'x_c3',
        #                   'lu00'.upper() :  'x_c3',
        #                   'lv00'.upper() :  'x_c3',
        #                   'me00'.upper() :  'x_c3',
        #                   'mk00'.upper() :  'x_c3',
        #                   'mt00'.upper() :  'x_c3',
        #                   'nl00'.upper() :  'x_c3',
        #                   'nos0'.upper() :  'x_c3',
        #                   'nom1'.upper() :  'x_c3',
        #                   'non1'.upper() :  'x_c3',
        #                   'pl00'.upper() :  'x_c3',
        #                   'pt00'.upper() :  'x_c3',
        #                   'ro00'.upper() :  'x_c3',
        #                   'rs00'.upper() :  'x_c3',
        #                   'se01'.upper() :  'x_c3',
        #                   'se02'.upper() :  'x_c3',
        #                   'se03'.upper() :  'x_c3',
        #                   'se04'.upper() :  'x_c3',
        #                   'si00'.upper() :  'x_c3',
        #                   'sk00'.upper() :  'x_c3',
        #                   'tr00'.upper() :  'x_c3',
        #                   'uk00'.upper() :  'x_c3',
        #                   'ukni'.upper() :	'x_c3'}        

        #%%
        for link in virtuallinksel.keys():
            try:
                out = robjects.r('''
                                createLink(
                                    from = "%s",
                                    to = "%s",
                                    propertiesLink = propertiesLinkOptions(
                                        hurdles_cost = FALSE,
                                        transmission_capacities = "enabled"
                                    ),
                                    dataLink = NULL
                                )
                                '''%(link, virtuallinksel[link]))
            except:
                print('Probably exist already. Trying to edit instead.')
                out = robjects.r('''
                                editLink(
                                    from = "%s",
                                    to = "%s",
                                    transmission_capacities = "enabled"
                                    )
                                '''%(link, virtuallinksel[link]))

        #%% Create binding constraints 

        start_number = 282
        areas = list(virtuallinksel.keys())

        for i in range(len(virtuallinksh2)):
            area = areas[i]

            print(
            '''
            [%d]
            name = CCGT_new_H2_%s
            id = ccgt_new_h2_%s
            enabled = true
            type = hourly
            operator = equal
            filter-year-by-year = 
            filter-synthesis = 
            z_h2_c3_%s%%z_taking = -0.580000
            %s.gas_ccgt new 2 = 1.000000
            '''%(start_number+i, area, area.lower(), area.lower(), area.lower())
            )
            
            # if 'ccgt_new_h2_%s.txt'%area.lower() not in os.listdir('Antares/input/bindingconstraints'):
            #     f = open('Antares/input/bindingconstraints/ccgt_new_h2_%s.txt'%area.lower(), 'w')
            #     f.close()
                
        for j in range(len(virtuallinksel)):
            area = areas[j]
            
            print(
            '''
            [%d]
            name = h2c3_eff_%s
            id = h2c3_eff_%s
            enabled = true
            type = hourly
            operator = equal
            filter-year-by-year = 
            filter-synthesis = 
            %s%%x_c3 = 0.700000
            x_c3%%z_h2_c3_%s = -1.000000
            '''%(start_number+i+j, area, area.lower(), area.lower(), area.lower())
            )
            
            # if 'h2c3_eff_%s.txt'%area.lower() not in os.listdir('Antares/input/bindingconstraints'):
            #     f = open('Antares/input/bindingconstraints/h2c3_eff_%s.txt'%area.lower(), 'w')
            #     f.close()

        #%% ...using antareseditobject

        import pickle
        with open('Pre-Processing/Output/BalmTechs.pkl', 'rb') as f:
            BalmTechs = pickle.load(f)
        with open('Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
            A2B_regi = pickle.load(f)
        with open('Pre-Processing/Output/B2A_regi.pkl', 'rb') as f:
            B2A_regi = pickle.load(f)
        with open('Pre-Processing/Output/B2A_DE_weights.pkl', 'rb') as f:
            B2A_DE_weights = pickle.load(f)

        fuelcons = {'MUNIWASTE' : 4}
        # fuelcons =    {'BIOGAS' : 2.13,
        #                'WOODCHIPS' : 3.34,
        #                'WOOD' : 3.34,
        #                'WOODWASTE' : 3.34,
        #                'PEAT' : 3.34,
        #                'STRAW' : 3.34}
        # Fixed efficiency and RHS values should be updated by Peri-Processing based on
        # residual load heuristic and average efficiency of the generator type fuelcons = 1/eff

        i = 0
        for area in A2B_regi.keys():
            print(
            """
            [%d]
            name = FlexDem_%s
            id = FlexDem_%s
            enabled = true
            type = weekly
            operator = greater
            filter-year-by-year = 
            filter-synthesis = 
            0_flexdem%%%s = -1
            0_flexdem.%s_flexloss = 1
            """%(644+i, area, area, area.lower(), area)
            )
            i += 1

            with open('0_flexdem_%s.txt'%area.lower(), 'w') as f:
                f.write('')



        #%% For muniwaste (region specific)
        for reg in A2B_regi.keys():
            for fuel in fuelcons.keys():
                try:
                    robjects.r('''
                            createBindingConstraint(
                                name = "{fuelup}RES_{reg}",
                                enabled = TRUE,
                                timeStep = "weekly",
                                operator = "greater",
                                coefficients = c("{reg}.condensing_{fuel}" = {fuelcons},
                                                "{reg}.chp-back-pressure_{fuel}" = {fuelcons},
                                                "{reg}.chp-extraction_{fuel}" = {fuelcons}),
                            )
                            '''.format(reg=reg, fuel=fuel, fuelup=fuel.upper(), fuelcons=fuelcons[fuel]))
                    # Remove binding constraint
                    # robjects.r('''
                    #         removeBindingConstraint(
                    #             name = "{fuelup}RES_{reg}",
                    #             )
                    #         '''.format(reg=reg, fuel=fuel, fuelup=fuel.upper(), fuelcons=fuelcons[fuel]))
                except:
                    print('{reg}-{fuel} constraint may already exist'.format(reg=reg, fuel=fuel))


        #%% For all other fuels (country specific)
        import gams
        import pandas as pd

        ws = gams.GamsWorkspace()
        ALLENDOFMODEL = ws.add_database_from_gdx(r'C:\\Users\\mberos\\gitRepos\\balmorel-antares' + '/Balmorel/base/model/all_endofmodel.gdx')
        CCCRRR = pd.DataFrame([rec.keys for rec in ALLENDOFMODEL['CCCRRR']], columns=['C', 'R']).groupby(by=['R']).aggregate({'C' : "sum"})

        fuelcons =    {'BIOGAS' : 2.13,
                    'WOODCHIPS' : 3.34,
                    'WOODPELLETS' : 3.34,
                    'WOOD' : 3.34,
                    'WOODWASTE' : 3.34,
                    'PEAT' : 3.34,
                    'STRAW' : 3.34}

        # for area in A2B_regi.keys():
        #     R = A2B_regi[area][0]
            
        #     # Amount of regions for that country
        #     N_reg = len(CCCRRR.loc[(CCCRRR.C == CCCRRR.loc[R, 'C'])])



        for fuel in fuelcons.keys():
            for country in CCCRRR.C.unique():
                coeffs = 'c('
                for R in CCCRRR.loc[CCCRRR.C == country].index:
                    for antarea in B2A_regi[R]:
                        for tech in BalmTechs.keys():
                            try:
                                BalmTechs[tech][fuel]
                                generator = '"{reg}.{tech}_{fuel}"'.format(reg=antarea.lower(), tech=tech.lower(), fuel=fuel.lower())
                                if not(generator in coeffs):
                                    coeffs += "{generator} = {fuelcons},\n".format(generator=generator, fuelcons=fuelcons[fuel]) 
                            except:
                                pass
                coeffs = coeffs.rstrip(',\n')
                coeffs += ')' 
                print('\n', coeffs)
                
                try:
                    robjects.r('''
                            createBindingConstraint(
                                name = "{fuelup}RES_{country}",
                                enabled = TRUE,
                                timeStep = "weekly",
                                operator = "less",
                                coefficients = {coeffs},
                            )
                            '''.format(country=country, fuelup=fuel.upper(), coeffs=coeffs))
                except:
                    pass
                
                
        #%% Actual H2 Transmission 
        links = [['AL', 'GR'],
                ['AL', 'MK'],
                ['AL', 'RS'],
                ['AT', 'DE4-S'],
                ['AT', 'IT'],
                ['AT', 'CH'],
                ['AT', 'CZ'],
                ['AT', 'SK'],
                ['AT', 'HU'],
                ['AT', 'SI'],
                ['BA', 'HR'],
                ['BA', 'RS'],
                ['BE', 'DE4-S'],
                ['BE', 'DE4-W'],
                ['BE', 'NL'],
                ['BE', 'UK'],
                ['BE', 'FR'],
                ['BE', 'LU'],
                ['BG', 'RO'],
                ['BG', 'GR'],
                ['BG', 'MK'],
                ['BG', 'RS'],
                ['BG', 'TR'],
                ['CH', 'DE4-S'],
                ['CH', 'FR'],
                ['CH', 'IT'],
                ['CH', 'AT'],
                ['CZ', 'DE4-E'],
                ['CZ', 'DE4-S'],
                ['CZ', 'PL'],
                ['CZ', 'AT'],
                ['CZ', 'SK'],
                ['DE4-E', 'DK2'],
                ['DE4-E', 'DE4-N'],
                ['DE4-E', 'DE4-S'],
                ['DE4-E', 'DE4-W'],
                ['DE4-E', 'SE4'],
                ['DE4-E', 'PL'],
                ['DE4-E', 'CZ'],
                ['DE4-N', 'DK1'],
                ['DE4-N', 'DE4-E'],
                ['DE4-N', 'DE4-W'],
                ['DE4-N', 'NO2'],
                ['DE4-N', 'SE4'],
                ['DE4-S', 'DE4-E'],
                ['DE4-S', 'DE4-W'],
                ['DE4-S', 'BE'],
                ['DE4-S', 'FR'],
                ['DE4-S', 'CH'],
                ['DE4-S', 'AT'],
                ['DE4-S', 'CZ'],
                ['DE4-S', 'LU'],
                ['DE4-W', 'DE4-E'],
                ['DE4-W', 'DE4-N'],
                ['DE4-W', 'DE4-S'],
                ['DE4-W', 'NL'],
                ['DE4-W', 'BE'],
                ['DE4-W', 'LU'],
                ['DK1', 'DK2'],
                ['DK1', 'DE4-N'],
                ['DK1', 'NL'],
                ['DK1', 'NO2'],
                ['DK1', 'SE3'],
                ['DK1', 'UK'],
                ['DK2', 'DK1'],
                ['DK2', 'DE4-E'],
                ['DK2', 'SE4'],
                ['EE', 'FIN'],
                ['EE', 'SE3'],
                ['EE', 'LV'],
                ['ES', 'FR'],
                ['ES', 'PT'],
                ['FIN', 'NO4'],
                ['FIN', 'SE1'],
                ['FIN', 'SE3'],
                ['FIN', 'EE'],
                ['FR', 'DE4-S'],
                ['FR', 'UK'],
                ['FR', 'BE'],
                ['FR', 'IT'],
                ['FR', 'CH'],
                ['FR', 'ES'],
                ['FR', 'LU'],
                ['GR', 'IT'],
                ['GR', 'BG'],
                ['GR', 'AL'],
                ['GR', 'MK'],
                ['GR', 'TR'],
                ['HR', 'HU'],
                ['HR', 'SI'],
                ['HR', 'BA'],
                ['HR', 'RS'],
                ['HU', 'AT'],
                ['HU', 'SK'],
                ['HU', 'SI'],
                ['HU', 'HR'],
                ['HU', 'RO'],
                ['HU', 'RS'],
                ['IE', 'UK'],
                ['LT', 'SE4'],
                ['LT', 'LV'],
                ['LT', 'PL'],
                ['LU', 'DE4-S'],
                ['LU', 'BE'],
                ['LU', 'FR'],
                ['LV', 'SE3'],
                ['LV', 'EE'],
                ['LV', 'LT'],
                ['MK', 'BG'],
                ['MK', 'GR'],
                ['MK', 'AL'],
                ['MK', 'RS'],
                ['NL', 'DK1'],
                ['NL', 'DE4-W'],
                ['NL', 'UK'],
                ['NL', 'BE'],
                ['NO1', 'NO2'],
                ['NO1', 'NO3'],
                ['NO1', 'NO5'],
                ['NO1', 'SE3'],
                ['NO2', 'DK1'],
                ['NO2', 'DE4-N'],
                ['NO2', 'NO1'],
                ['NO2', 'NO5'],
                ['NO2', 'UK'],
                ['NO3', 'NO1'],
                ['NO3', 'NO4'],
                ['NO3', 'NO5'],
                ['NO3', 'SE2'],
                ['NO4', 'FIN'],
                ['NO4', 'NO3'],
                ['NO4', 'SE1'],
                ['NO4', 'SE2'],
                ['NO5', 'NO1'],
                ['NO5', 'NO2'],
                ['NO5', 'NO3'],
                ['PL' , 'DE4-E'],
                ['PL' , 'SE4'],
                ['PL' , 'LT'],
                ['PL' , 'CZ'],
                ['PL' , 'SK'],
                ['PT' , 'ES'],
                ['RO' , 'HU'],
                ['RO' , 'BG'],
                ['RS' , 'HU'],
                ['RS' , 'HR'],
                ['RS' , 'BG'],
                ['RS' , 'AL'],
                ['RS' , 'MK'],
                ['RS' , 'BA'],
                ['SE1' , 'FIN'],
                ['SE1' , 'NO4'],
                ['SE1' , 'SE2'],
                ['SE2' , 'NO3'],
                ['SE2' , 'NO4'],
                ['SE2' , 'SE1'],
                ['SE2' , 'SE3'],
                ['SE3' , 'DK1'],
                ['SE3' , 'FIN'],
                ['SE3' , 'NO1'],
                ['SE3' , 'SE2'],
                ['SE3' , 'SE4'],
                ['SE3' , 'EE'],
                ['SE3' , 'LV'],
                ['SE4' , 'DK2'],
                ['SE4' , 'DE4-E'],
                ['SE4' , 'DE4-N'],
                ['SE4' , 'SE3'],
                ['SE4' , 'LT'],
                ['SE4' , 'PL'],
                ['SI' , 'IT'],
                ['SI' , 'AT'],
                ['SI' , 'HU'],
                ['SI' , 'HR'],
                ['SK' , 'PL'],
                ['SK' , 'AT'],
                ['SK' , 'CZ'],
                ['SK' , 'HU'],
                ['TR' , 'BG'],
                ['TR' , 'GR'],
                ['UK' , 'DK1'],
                ['UK' , 'NL'],
                ['UK' , 'NO2'],
                ['UK' , 'BE'],
                ['UK' , 'FR'],
                ['UK' , 'IE']]



        import pickle
        with open('Pre-Processing/Output/B2A_regi_h2.pkl', 'rb') as f:
            B2A_regi_h2 = pickle.load(f) 
            
        ## Assuming only 1-1 spatial resoluted nodes (Italy is taken care of)
        chosenlinks = []
        for link in links:
            # try:
            #     out = robjects.r('''
            #                     createLink(
            #                         from = "%s",
            #                         to = "%s",
            #                         propertiesLink = propertiesLinkOptions(
            #                             hurdles_cost = FALSE,
            #                             transmission_capacities = "enabled"
            #                         ),
            #                         dataLink = NULL
            #                     )
            #                     '''%(B2A_regi_h2[link[0]][0], B2A_regi_h2[link[1]][0]))
            # except:
            try:
                print('Probably exist already. Trying to edit instead.')
                out = robjects.r('''
                                editLink(
                                    from = "%s",
                                    to = "%s",
                                    transmission_capacities = "enabled"
                                    )
                                '''%(B2A_regi_h2[link[0]][0], B2A_regi_h2[link[1]][0]))
                
                # Don't append if it's already defined in other direction
                if not([link[1], link[0]] in chosenlinks):
                    chosenlinks.append(link)
            except:
                pass

        #%%
        for link in chosenlinks:
            print('%s %s'%(B2A_regi_h2[link[0]][0], B2A_regi_h2[link[1]][0]))
            # print('%s %s'%(link[0], link[1]))

        #%% 2.2 How to create binding constraints

        # createBindingConstraint(
        #   name = "myconstraint", 
        #   values = matrix(data = c(rep(c(19200, 0, 0), each = 366)), ncol = 3), 
        #   enabled = FALSE, 
        #   timeStep = "daily",
        #   operator = "both",
        #   coefficients = c("fr%myarea" = 1)
        # )

        #%% 2.3 How to create DSR

        # dsrData <- data.frame(
        #   area = c("a", "b"),
        #   unit = c(10,20), 
        #   nominalCapacity = c(100, 120),
        #   marginalCost = c(52, 65),
        #   hour = c(3, 7)
        # )
        
        # createDSR(dsrData)

        #%% 2.4 Creating VRE Clusters

        import pickle
        with open('Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
            A2B_regi = pickle.load(f)
            
        for area in list(A2B_regi.keys()):
            for sres in ['_0', '_5', '_6', '_7', '_8']:
                out = robjects.r('''
                            createClusterRES(
                                area = "%s",
                                cluster_name = "%s",
                                group = "Wind Onshore",
                                add_prefix = FALSE,
                                `ts-interpretation` = "production-factor"
                                )
                            '''%(area, area + "_wind" + sres))
                out = robjects.r('''
                            createClusterRES(
                                area = "%s",
                                cluster_name = "%s",
                                group = "Solar PV",
                                add_prefix = FALSE,
                                `ts-interpretation` = "production-factor"
                                )
                            '''%(area, area + "_solar" + sres))

                # How to update .ini files (forgot the nominalcapacity, unitcount and enabled options):
                # Config2 = configparser.ConfigParser()
                # Config2.read('../Antares/input/renewables/clusters/%s/list.ini'%area)
                # Config2.set(area + "_wind" + sres, 'nominalcapacity', '0.0')
                # Config2.set(area + "_wind" + sres, 'unitcount', '1')
                # Config2.set(area + "_wind" + sres, 'enabled', 'false')
                # Config2.set(area + "_solar" + sres, 'nominalcapacity', '0.0')
                # Config2.set(area + "_solar" + sres, 'unitcount', '1')
                # Config2.set(area + "_solar" + sres, 'enabled', 'false')
                # with open('../Antares/input/renewables/clusters/%s/list.ini'%area, 'w') as file:
                #     Config2.write(file)

        #%% Creating thermal clusters
        import pickle
        with open('Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
            A2B_regi = pickle.load(f)
        with open('Pre-Processing/Output/BalmTechs.pkl', 'rb') as f:
            BalmTechs = pickle.load(f)
            
        for area in A2B_regi.keys():
            # for tech in ['CHP-BACK-PRESSURE', 'CHP-EXTRACTION', 'CONDENSING']:
            #     for fuel in ['WOOD', 'WOODWASTE', 'PEAT', 'STRAW']:
            

            # Create
            try:
                # out = robjects.r('''
                #     createCluster(
                #         area = "0_flexdem",
                #         cluster_name = "{area}_flexloss",
                #         group = "other",
                #         enabled = TRUE,
                #         add_prefix = FALSE,
                #         `unitcount` = 1, 
                #         `nominalcapacity` = 500000,
                #         `marginal-cost` = 3000,
                #         `market-bid-cost` = 3000
                #         )
                #     '''.format(area=area))
                    # '''.format(area=area.lower(), name='_'.join((tech, fuel)).lower(), fuel=fuel.lower()))
            
                with open(r'C:\Users\mberos\gitRepos\balmorel-antares\Antares\input\thermal\series\0_flexdem\%s_flexloss\series.txt'%area.lower(), 'w') as f:
                    for i in range(8760):
                        f.write('500000\n')
            except Exception as e:
                raise e
                pass            
            # Remove
            # out = robjects.r('''
            #     removeCluster(
            #         area = "{area}",
            #         cluster_name = "{name}",
            #         add_prefix=FALSE
            #         )
            #     '''.format(area=area, name='_'.join((tech, fuel))))
                    
        #%% 2.5 Remove Clusters

        import pickle
        with open('Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
            A2B_regi = pickle.load(f)

        for VRE in ['wind', 'solar']:
            for area in A2B_regi.keys():
                for sres in ['_5', '_6', '_7', '_8']:
                    out = robjects.r('''
                                removeClusterRES("%s", "%s", add_prefix=FALSE)
                                '''%(area, area + "_%s"%VRE + sres))
                # out = robjects.r('''
                #             createClusterRES(
                #                 area = "%s",
                #                 cluster_name = "%s",
                #                 group = "Solar PV",
                #                 add_prefix = FALSE,
                #                 `ts-interpretation` = "production-factor"
                #                 )
                #             '''%(area, area + "_solar" + sres))


        #%%
        ### 2.4 AntaresViz
        # importr("antaresViz")
        # Note that savePlotAsPng requires
        # "PhantomJS" which can be installed
        # by executing: webshot::install_phantomjs()

        ## Hourly Profile
        # robjects.r('''
        #         setSimulationPath(%r, %r)
        #         mydata <- readAntares()
                
        #         # Electricity
        #         fplot  <- prodStack(mydata, main="%s",
        #                     dateRange=c("%s", "%s"),  
        #                     areas=c("de00", "dke1", "dkw1"),
        #                     interactive=FALSE)
        #         savePlotAsPng(fplot, file="OverallResults/%s.png", width=800, height=500) # <- note that this requires the PhantomJS
                
        #         # Hydrogen
        #         fplot  <- prodStack(mydata, main="%s",
        #                     dateRange=c("%s", "%s"),  
        #                     areas=c("z_h2_c3_de00", "z_h2_c3_dke1", "z_h2_c3_dkw1"),
        #                     interactive=FALSE)
        #         savePlotAsPng(fplot, file="OverallResults/%s.png", width=800, height=500) # <- note that this requires the PhantomJS
        #         '''%tuple([wk_dir + '/Antares'] + 2*[SC + '_Iter%d_Y-%s'%(i, year)] + [startDate, endDate] + [SC + '_Iter%d_Y-%s_elechourly'%(i, year)] +\
        #             [SC + '_Iter%d_Y-%s'%(i, year)] + [startDate, endDate] + [SC + '_Iter%d_Y-%s_h2hourly'%(i, year)]))                        

        # ## Exchanges (negative is import)
        # robjects.r('''
        #             fplot  <- exchangesStack(mydata, 
        #                                     area = "%s", 
        #                                     main = "Import/Export of %s", 
        #                                     unit = "GWh", 
        #                                     interactive = FALSE)
        #             savePlotAsPng(fplot, file="OverallResults/%s.png", width=600, height=1000) # <- note that this requires the PhantomJS
        #             '''%(area, area, SC + '_Iter%d_Y-%s_exchanges-%s'%(i, year, area)))
