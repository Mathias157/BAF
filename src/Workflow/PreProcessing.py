"""
Created on 09-06-2023

@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###          Script Settings        ###
### ------------------------------- ###

import click
import pandas as pd
from pandas.errors import EmptyDataError
import os
import sys
sys.path.append('.')
from Functions.GeneralHelperFunctions import load_OSMOSE_data, data_context
from pybalmorel import IncFile
import pickle
import configparser

# Load geojson's
try:
    import geopandas as gpd
    p = 'Pre-Processing/Data/Balmorelgeojson'
    
    if 'balmmap' not in locals():
        balmmap = gpd.read_file('Pre-Processing/Data/231206 AntBalmMap.geojson')
    plot_maps = True
        
except ModuleNotFoundError:
    plot_maps = False
    print('Geopandas not installed. Will not be able to plot maps')
    balmmap = pd.DataFrame({'id' : 'a'},index=[0])

@click.group()
@click.pass_context
def CLI(ctx):
    """The command line interface for pre-processing stuff"""

    ### 0.0 Load configuration file
    Config = configparser.ConfigParser()
    Config.read('Config.ini')
    balmorel_weather_year = Config.getint('PreProcessing', 'balmorel_weather_year')

    # Set geographical scope
    ctx.ensure_object(dict)
    ctx.obj['geographical_scope'] = Config.get('PreProcessing', 'geographical_scope').replace(' ', '').split(',')
    
    # Set Balmorel weather year choice
    ctx.obj['balmorel_weather_year'] = balmorel_weather_year
    
    # Detect which command has been passed
    command = ctx.invoked_subcommand
    
    ## Define Balmorel time index
    if command in ['generate-antares-vre', 'generate-balmorel-hydro', 'generate-balmorel-timeseries', 'generate-balmorel-heat-series']:
        # Create S and T timeseries
        ctx.obj['S'] = ['S0%d'%i for i in range(1, 10)] + ['S%d'%i for i in range(10, 53)]
        ctx.obj['T'] = ['T00%d'%i for i in range(1, 10)] + ['T0%d'%i for i in range(10, 100)] + ['T%d'%i for i in range(100, 169)]
        ctx.obj['ST'] = [S + ' . ' + T for S in ctx.obj['S'] for T in ctx.obj['T']]
        
    ## Set paths for data
    if command in ['generate-antares-vre', 'generate-balmorel-timeseries', 'generate-balmorel-heat-series', 'generate-balmorel-annual-heat-adjustment']:
        data_context()

    ## Set weather years
    if command in ['generate-balmorel-timeseries', 'generate-balmorel-heat-series', 'generate-balmorel-hydro-series']:
        ctx.obj['weather_years'] = [balmorel_weather_year]
    elif command in ['generate-antares-vre', 'generate-balmorel-annual-heat-adjustment']:
        ctx.obj['weather_years'] = [1982, 1983, 1984, 1985, 1986, 
                                    1987, 1988, 1989, 1990, 1991, 
                                    1992, 1993, 1994, 1995, 1996, 
                                    1997, 1998, 1999, 2000, 2001, 
                                    2002, 2003, 2004, 2005, 2006, 
                                    2007, 2008, 2009, 2010, 2011, 
                                    2012, 2013, 2014, 2015, 2016]


### ------------------------------- ###
###            Utilities            ###
### ------------------------------- ###
def read_incfile_presuf(name: str, 
                      weather_year: int,
                      incfile_prefix_path: str = 'Pre-Processing/Data/IncFile PreSuffixes',
                      presuf: str = 'pre'):
    """Reads an .inc file that should just contain the prefix of the incfile

    Args:
        name (str): Name of the incfile
        incfile_prefix_path (str): Path to where the related prefix .inc file is
        weather_year (int): The weather year, which is relevant for the description of some .inc files.

    Returns:
        _type_: _description_
    """
    # Add information about weather year if weather dependent parameter
    if ('WND' in name) | ('SOLE' in name) | ('WTR' in name) | ('DH' in name and not ('INDUSTRY' in name or 'HYDROGEN' in name)):
        string = "* Weather year %d\n"%(weather_year) + ''.join(open(incfile_prefix_path + '/%s.inc'%name).readlines())
    else:
        string = ''.join(open(incfile_prefix_path + '/%s.inc'%name).readlines())

    # Get prefix or suffix
    if presuf == 'pre':
        string = string.split('*PRESUFSPLIT*')[0]
    else:
        try:
            string = string.split('*PRESUFSPLIT*')[1]
        except IndexError:
            raise IndexError("%s didn't include a *PRESUFSPLIT* line, making it impossible to figure out what the suffix is!"%name)    
    
    return string

def create_incfiles(names: list, 
                    weather_year: int,
                    bodies: dict = None, 
                    suffixes: dict = None):
    """A convenient way to create many incfiles and fill in predefined prefixes, bodies and suffixes

    Args:
        names (list): Names of the incfiles created (without .inc)
        weather_year (int): The weather year for weather dependent parameters.
        bodies (dict, optional): bodies of the incfiles. Defaults to None.
        suffixes (dict, optional): suffixes of the incfiles. Defaults to None.
        incfile_prefix_path (str, optional): path to the prefixes of the incfiles. Defaults to 'Pre-Processing/Data/IncFile Prefixes'.
        
    Returns:
        incfiles (dict): Dictionary of incfiles
    """
    
    incfiles = {
        incfile : IncFile(name=incfile, prefix=read_incfile_presuf(incfile, weather_year), suffix=read_incfile_presuf(incfile, weather_year, presuf='suf')) \
        for incfile in names
    }
    
    if bodies is not None:
        for incfile in bodies.keys():
            incfiles[incfile].body = bodies[incfile]
            
    return incfiles

def append_neighbouring_years(filename: str, year: int, values: str,
                              index: str = 'timestamp', columns: str = 'country'):
    """Include year before and after for year-separated .csv files"""
    df = (
        pd.read_csv(filename)
        .pivot_table(index=index, columns=columns, values=values, aggfunc='sum',
                     fill_value=0)   
    )
    
    for neighbour_year in [year-1, year+1]:
        neighbour_filename = filename.replace(str(year), str(neighbour_year))
        # Check if file for neighbouring year exists
        try:
            temp = (
            pd.read_csv(neighbour_filename)
            .pivot_table(index=index, columns=columns, 
                         values=values, aggfunc='sum',
                        fill_value=0)   
            )
            
            # Remove most of the year before and after, except one week
            if neighbour_year == year - 1:
                temp = temp.iloc[-168:]
                df = pd.concat((temp, df))
            elif neighbour_year == year + 1:
                temp = temp.iloc[:168]
                df = pd.concat((df, temp))
            else:
                raise ValueError('Something is wrong in finding neighbouring years!')

        
        except FileNotFoundError:
            print("Couldn't find year %d of file %s"%(neighbour_year, neighbour_filename))

    return df
    
@click.pass_context
def convert_to_52weeks(ctx, year: int, filename: str, values: str):
    """Convert timeseries to start on first hour of first monday and be 52 weeks long"""
    
    filename = filename%year
    df = pd.read_csv(filename).pivot_table(index='time_id', columns='country', values=values)
    
    # Just take first 52 weeks
    df = df.iloc[:8736]
    
    # Apply S and T index
    df.index = ctx.obj['ST']
    
    return df

def list_of_str(value):
    return value.replace(' ', '').split(',')


#%% ------------------------------- ###
###        Hardcoded Mappings       ###
### ------------------------------- ###

@CLI.command()
@click.pass_context
def generate_mappings(ctx):
    """Generates spatial and technological mappings between Balmorel and Antares"""
    
    ### 1.1 Regions for A2B2A Mapping
    geographical_scope = ctx.obj['geographical_scope']
    
    # Regions for VRE Mapping (currently uniform mapping)
    B2A_regi = {region : [region] for region in geographical_scope}

    A2B_regi = {region : [region] for region in geographical_scope}


    ## Save defined regions
    with open('Pre-Processing/Output/B2A_regi.pkl', 'wb') as f:
        pickle.dump(B2A_regi, f)
    with open('Pre-Processing/Output/A2B_regi.pkl', 'wb') as f:
        pickle.dump(A2B_regi, f)
        
        
    ### 1.2 Other useful dictionaries

    # Balmorel Technologies (rough combustion CO2 estimates hardcoded for now)
    kgGJ2tonMWh = 3.6 / 1e3 # Conversion from kg/GJ to ton/MWh
    BalmTechs = {'CHP-BACK-PRESSURE' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh}, 
                                        'WOODCHIPS' : {'CO2' : 0},  
                                        'WOODPELLETS' : {'CO2' : 0},  
                                        'STRAW' : {'CO2' : 0}, 
                                        'BIOGAS' : {'CO2' : 0}, 
                                        'MUNIWASTE' : {'CO2' : 0},
                                        'COAL' : {'CO2' : 94.6*kgGJ2tonMWh},
                                        'FUELOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                        'LIGHTOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                        'LIGNITE' : {'CO2' : 111.1*kgGJ2tonMWh}},
                'CHP-EXTRACTION' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh}, 
                                    'WOODCHIPS' : {'CO2' : 0},  
                                    'WOODPELLETS' : {'CO2' : 0}, 
                                    'BIOGAS' : {'CO2' : 0}, 
                                    'COAL' : {'CO2' : 94.6*kgGJ2tonMWh},
                                    'LIGNITE' : {'CO2' : 111.1*kgGJ2tonMWh},
                                    'MUNIWASTE' : {'CO2' : 0},
                                    'WOODWASTE' : {'CO2' : 0},
                                    'WOOD' : {'CO2' : 0},
                                    'PEAT' : {'CO2' : 0},
                                    'STRAW' : {'CO2' : 0}},
                'CONDENSING' : {'BIOGAS' : {'CO2' : 0}, 
                                'COAL' : {'CO2' : 94.6*kgGJ2tonMWh},
                                'FUELOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                'LIGHTOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                'LIGNITE' : {'CO2' : 111.1*kgGJ2tonMWh},
                                'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh}, 
                                'WOODCHIPS' : {'CO2' : 0}, 
                                'MUNIWASTE' : {'CO2' : 0},
                                'NUCLEAR' : {'CO2' : 0},
                                'STRAW' : {'CO2' : 0},
                                'WOOD' : {'CO2' : 0},
                                'PEAT' : {'CO2' : 0}},
                'CONDENSING-CCS' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh * 0.1}}, 
                'CHP-BACK-PRESSURE-CCS' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh * 0.1}}, 
                'CHP-EXTRACTION-CCS' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh * 0.1}}, 
                'FUELCELL' : {'HYDROGEN' : {'CO2' : 0}}}
    {'BIOGAS' : 2.13,
                'WOODCHIPS' : 3.34,
                'MUNIWASTE' : 4,
                'WOOD' : 3.34,
                'WOODWASTE' : 3.34,
                'PEAT' : 3.34,
                'STRAW' : 3.34}

    with open('Pre-Processing/Output/BalmTechs.pkl', 'wb') as f:
        pickle.dump(BalmTechs, f)

#%% ------------------------------- ###
###            Timeseries           ###
### ------------------------------- ###


@CLI.command()
@click.pass_context
@load_OSMOSE_data(files=['offshore_wind', 'onshore_wind', 'solar_pv'])
def generate_antares_vre(ctx, data: str, stoch_year_data: dict, antares_input_paths = {
        'offshore_wind' : 'Antares/input/renewables/series/%s/offshore/series.txt',
        'onshore_wind' : 'Antares/input/renewables/series/%s/onshore/series.txt',
        'solar_pv' : 'Antares/input/renewables/series/%s/photovoltaics/series.txt',
    }):
    """Generate production factor timeseries for Antares VRE"""
    
    group_name = {'offshore_wind' : 'Wind Offshore',
                  'onshore_wind' : 'Wind Onshore',
                  'solar_pv' : 'Solar PV'}
    folder_name = {'offshore_wind' : 'offshore',
                  'onshore_wind' : 'onshore',
                  'solar_pv' : 'photovoltaics'}
    
    
    # Create matrix of input that Antares expects
    for region in ctx.obj['geographical_scope']:
        
        try:   
            data_to_antares_input = pd.DataFrame({year : stoch_year_data[year][region] for year in stoch_year_data.keys()})
            
            # Make series input
            series_folder_path = (antares_input_paths[data]%region.lower()).replace('/series.txt', '')
            if not(os.path.exists(series_folder_path)):
                os.makedirs(series_folder_path)
            
            data_to_antares_input.to_csv(antares_input_paths[data]%(region.lower()),
                        index=False, header=False, sep='\t')
            
            # Make .ini file
            ini_folder_path = series_folder_path.replace(f'series/{region.lower()}/{folder_name[data]}', f'clusters/{region.lower()}')
            if not(os.path.exists(ini_folder_path)):
                os.makedirs(ini_folder_path)
                file = open('list.ini', 'w')
                file.close()
                
            inifile=configparser.ConfigParser()
            inifile.read(ini_folder_path + '/list.ini')
            try:
                inifile.add_section(folder_name[data])
                inifile.set(folder_name[data], 'name', folder_name[data])
                inifile.set(folder_name[data], 'group', group_name[data])
                inifile.set(folder_name[data], 'nominalcapacity', '0')
                inifile.set(folder_name[data], 'unitcount', '1')
                inifile.set(folder_name[data], 'ts-interpretation', 'production-factor')
                with open(ini_folder_path + '/list.ini', 'w') as f:
                    inifile.write(f)
            except configparser.DuplicateSectionError:
                print(f'{data} already exists in ini file for {region}')
            
        except KeyError:
            print('No %s for %s'%(data, region))
    
    
@CLI.command()
@click.pass_context   
@load_OSMOSE_data(files=['offshore_wind', 'onshore_wind', 'solar_pv', 'load']) 
def generate_balmorel_timeseries(ctx, data: str, stoch_year_data: dict):
    """Generate Balmorel timeseries input data for VRE (except hydro) and exogenous electricity demand"""
    
    # Format data to Balmorel input
    balmorel_names = {
        'offshore_wind' : {'incfile' : 'WND_VAR_T_OFF',
                           'area_suffix' : '_OFF'}, 
        'onshore_wind'  : {'incfile' : 'WND_VAR_T',
                           'area_suffix' : '_A'},
        'solar_pv'      : {'incfile' : 'SOLE_VAR_T',
                           'area_suffix' : '_A'},
        'load'          : {'incfile' : 'DE_VAR_T',
                           'area_suffix' : ''}
    }
    
    # Format data
    ## Only the chosen weather year for Balmorel is loaded here 
    year_data_column = list(stoch_year_data.keys())[0] 
    df = stoch_year_data[year_data_column]
    FLH = df.sum()
    df = df.loc[:8736, :] # Just leave out the last (two last) day(s)
    df.index = ctx.obj['ST']  
    df.columns = pd.Series(df.columns) + balmorel_names[data].get('area_suffix')
    df.columns.name = ''
            
    # Create .inc files for Balmorel
    f = IncFile(name=balmorel_names[data].get('incfile'),
                prefix=read_incfile_presuf(balmorel_names[data].get('incfile'), ctx.obj['balmorel_weather_year']),
                suffix=read_incfile_presuf(balmorel_names[data].get('incfile'), ctx.obj['balmorel_weather_year'], presuf='suf'))
    f.body = df
    f.save()

    if data != 'load':
        FLH_name = balmorel_names[data].get('incfile').replace('_VAR_T', 'FLH')
        f = IncFile(name=FLH_name,
                    prefix=read_incfile_presuf(FLH_name, ctx.obj['balmorel_weather_year']),
                    suffix=read_incfile_presuf(FLH_name, ctx.obj['balmorel_weather_year'], presuf='suf'),
                    body="\n".join(['%s%s %0.2f'%(col, balmorel_names[data]['area_suffix'], FLH[col]) for col in FLH.index]))
        f.save()


@CLI.command()
@click.pass_context
@load_OSMOSE_data(files=['heat'])
def generate_balmorel_heat_series(ctx, data: str, stoch_year_data: dict):
    # Format data
    weather_year = ctx.obj['balmorel_weather_year']
    df = stoch_year_data[weather_year]
    df = df.loc[:8736, :] # Just leave out the last (two last) day(s)
    df.index = ctx.obj['ST']  
    df.columns.name = ''

    # Individual heat 
    df.columns = pd.Series(df.columns) + '_IDVU-SPACEHEAT'
     
    # Make constant, 20% heat for hot water assumption in district heat
    df_dh = 0.2/8736 + df*0.8
    
    # District Heat
    df_dh.columns = pd.Series(df_dh.columns).str.replace('_IDVU-SPACEHEAT', '_A')

    incfiles = create_incfiles(names=['DH_VAR_T', 'INDIVUSERS_DH_VAR_T'],
                               weather_year=weather_year,
                               bodies={'DH_VAR_T' : df_dh,
                                       'INDIVUSERS_DH_VAR_T' : df})

    for incfile in incfiles.values():
        incfile.save()

@CLI.command()
@click.pass_context
@load_OSMOSE_data(files=['heat'])
def generate_balmorel_annual_heat_adjustment(ctx, data: str, stoch_year_data: dict):
    
    # Get factors on annual heat demand
    weather_year = ctx.obj['balmorel_weather_year']
    factors, normal_year = get_annual_heat_demand_factor(stoch_year_data, weather_year)
    
    # Add the factor commands to DH.inc and INDIVUSERS_DH.inc 
    DH_COMMANDS = f'* Weather year {weather_year} factors relative to normal weather year {normal_year}, based on sum of heat demand throughout regions\n'
    INDIVDH_COMMANDS = f'* Weather year {weather_year} factors relative to normal weather year {normal_year}, based on sum of heat demand throughout regions\n'
    
    for region in factors.index:
        # 80% of district heating demand is space heating
        DH_COMMANDS += f"DH(YYY, AAA, DHUSER)$RRRAAA('{region}', AAA) = 0.8*{factors[region]}*DH(YYY, AAA, DHUSER)$RRRAAA('{region}', AAA) + 0.2*DH(YYY, AAA, DHUSER)$RRRAAA('{region}', AAA);\n"
        
        # Choosing only space heat individual heating areas
        INDIVDH_COMMANDS += f"DH(YYY, '{region}_IDVU-SPACEHEAT', 'TERTIARY'   ) = {factors[region]}*DH(YYY, '{region}_IDVU-SPACEHEAT', 'TERTIARY'   );\n"
        INDIVDH_COMMANDS += f"DH(YYY, '{region}_IDVU-SPACEHEAT', 'RESIDENTIAL') = {factors[region]}*DH(YYY, '{region}_IDVU-SPACEHEAT', 'RESIDENTIAL');\n"

    replace_text_in_file('Balmorel/base/data/DH.inc', DH_COMMANDS)
    replace_text_in_file('Balmorel/base/data/INDIVUSERS_DH.inc', INDIVDH_COMMANDS)

    # Output a random file in order to connect snakemake rules
    with open('Balmorel/base/data/done.txt', 'w') as f:
        f.write('finished modifying DH.inc and INDIVUSERS_DH.inc')

def replace_text_in_file(file: str, text: str):
    """Will add or replace content of a text file within specific modification warning labels

    Args:
        file (str): The file to append or replace content within warning labels
        text (str): The text to put within warning labels
    """
    
    # Factor district heat ares by 0.8*factors
    with open(file, 'r') as f:
        content = f.read()
    
    ## Remove previous insertions
    before_label = '* ------ DO NOT MODIFY THIS LINE OR BELOW ------'
    after_label = '* ------ DO NOT MODIFY THIS LINE OR ABOVE ------'
    if before_label in content and after_label in content:
        content_cleaned = content.split(before_label)[0] + content.split(after_label)[1]
    else:
        content_cleaned = content
    
    ## Write new file with content appended to bottom
    with open(file, 'w') as f:
        f.write(content_cleaned)
        f.write('\n')
        f.write(before_label)
        f.write('\n')
        f.write(text)
        f.write('\n')
        f.write(after_label)
        f.write('\n')
        

def get_annual_heat_demand_factor(stoch_year_data: dict, weather_year: int):
    """Finds median annual heat demand year and returns the factor
    calculated relative to the median year, that should be 
    multiplied to annual heat demand per region

    Args:
        stoch_year_data (dict): The heat data
        weather_year (int): The chosen weather year

    Returns:
        factors (pd.Series): Factors on chosen weather year per region
        most_normal_year (int): The normal weather year
    """
    
    data = []
    for year in stoch_year_data.keys():
        data.append(
            stoch_year_data[year].sum().to_dict() | {'Weather Year' : year}
        )
        
    df = pd.DataFrame(data).pivot_table(index='Weather Year')
    idx_of_most_normal_year = df.sum(axis=1).median() == df.sum(axis=1)
    most_normal_year = df[idx_of_most_normal_year].index[0] # Just take the first one, if there were identical years

    factors = df.loc[weather_year] / df.loc[most_normal_year]
         
    return factors, most_normal_year

### ------------------------------- ###
###            Hydropower           ###
### ------------------------------- ###
@CLI.command()
@click.pass_context
def generate_balmorel_hydro(ctx):
    """Generate Balmorel input data for hydropower"""

    # Read parameters
    ## E.g., 30 = 2012, note we start counting from 0 here, compared to in Antares UI!
    weather_year = ctx.obj['balmorel_weather_year'] - 1982

    ## Balmorel timestamps
    S = ctx.obj['S']
    ST = ctx.obj['ST']

    # Prepare placeholders
    hydro_res = configparser.ConfigParser()
    hydro_res.read('Antares/input/hydro/hydro.ini')
    GKFX_hydro_split_string_start = '* ---- START OF HYDRO HARMONISATION - DONT REMOVE OR MODIFY THIS LINE ----'
    GKFX_hydro_split_string_end = '* ---- END OF HYDRO HARMONISATION - DONT REMOVE OR MODIFY THIS LINE ----'
    with open('Balmorel/base/data/GKFX.inc', 'r') as f:
        GKFX = f.read()
        GKFX_start = GKFX.split(GKFX_hydro_split_string_start)[0] + GKFX_hydro_split_string_start + '\n'
        GKFX_end = GKFX.split(GKFX_hydro_split_string_end)[1] + GKFX_hydro_split_string_end + '\n'
        
    hydro_AAA = '\n'
    # GNR_RES_WTR_NOPMP (100 % efficiency)
    # GNR_RES_WTR_PMP_MC-01 (100 % efficiency)
    hydro_GKFX = '\n'
    hydro_WTRRRFLH = ''
    hydro_WTRRSFLH = ''
    hydro_WTRRSVAR_S = pd.DataFrame(index=S)
    hydro_WTRRRVAR_T = pd.DataFrame(index=ST)
    hydro_HYRSMAXVOL_G = '\n'

    for area in ctx.obj['geographical_scope']:
        print('Generating hydro data for %s'%area)
        ### Reservoir power capacity and pumping eff in the area itself
        try:
            turb_cap = pd.read_table('Antares/input/hydro/common/capacity/maxpower_%s.txt'%area.lower(), header=None)[0].max() # MW
            pump_cap = pd.read_table('Antares/input/hydro/common/capacity/maxpower_%s.txt'%area.lower(), header=None)[2].max() # MW
            
            res_series = pd.read_table('Antares/input/hydro/series/%s/mod.txt'%area.lower(), header=None) # MW

            # Summing weekly flows for Balmorel
            hydro_WTRRSVAR_S['%s_A'%area] = res_series.rolling(window=7).sum()[6::7][weather_year].values
            
            # FLH
            hydro_WTRRSFLH += "%s_A\t\t%0.2f\n"%(area, ((res_series[weather_year]).sum())/(8760*turb_cap)*8760)
              
            # Make .inc file commands
            hydro_AAA += '%s_A\n'%area 
            if pump_cap == 0:
                G = 'GNR_RES_WTR_NOPMP'
                hydro_GKFX += "GKFX(YYY,'%s_A','%s') = %0.2f;\n"%(area, G, turb_cap)
            else:
                G = 'GNR_RES_WTR_PMP_MC-01'
                hydro_GKFX += "GKFX(YYY,'%s_A','%s') = %0.2f;\n"%(area, G, turb_cap)
            
            print('Found reservoir data..')
            
            # See if there's a reservoir capacity
            try:
                res_cap = hydro_res.getfloat('reservoir capacity', area)
                hydro_HYRSMAXVOL_G += "HYRSMAXVOL_G('%s_A', '%s') = %0.2f;\n"%(area, G, res_cap/turb_cap)
                print('..with pump')
            except configparser.NoOptionError:
                hydro_HYRSMAXVOL_G += "HYRSMAXVOL_G('%s_A', '%s') = %0.2f;\n"%(area, G, 0)
            
        except EmptyDataError:
            # No reservoir storage in area
            pass

        ### Run of river in the area itself
        try:
            ror_series = pd.read_table('Antares/input/hydro/series/%s/ror.txt'%area.lower(), header=None) # MW
            ror_cap = ror_series.max().max()

            # Make .inc file commands        
            if '%s_A'%area not in hydro_AAA:
                hydro_AAA += '%s_A\n'%area 
                
            hydro_WTRRRVAR_T['%s_A'%area] = ror_series.loc[:8735, weather_year].values
            
            hydro_WTRRRFLH += "%s_A\t\t%0.2f\n"%(area, ((ror_series[weather_year]).sum())/(8760*ror_cap)*8760)

            
            hydro_GKFX += "GKFX(YYY, '%s_A', 'GNR_ROR_WTR') = %0.2f;\n"%(area, ror_cap)
            print('Found run-of-river data..')
        except EmptyDataError:
            pass

        ### Pumped Hydro Storage (No inflow) 
        for hydro_area in ['00_PSP_STO']:
            
            try:
                turb_cap = pd.read_table('Antares/input/links/%s/capacities/%s_direct.txt'%(hydro_area, area.lower()), header=None).max()[0]
                
                # .inc file commands
                # Note that this capacity is defined in MWh! Thus factored the unloading parameter GDSTOHUNLD of  9.4 
                hydro_GKFX += "GKFX(YYY, '%s_A', 'GNR_ES_WTR_PMP') = %0.2f;\n"%(area, turb_cap*9.4)
                print('Found pumped hydro data..')

            except FileNotFoundError:
                pass
            
            #'GNR_ES_WTR_PMP' changed to 75% efficiency in GKFX incfile prefix

    # Create 
    incfiles = create_incfiles(['WTRRSVAR_S', 'WTRRSFLH', 'WTRRRVAR_T', 'WTRRRFLH', 'HYRSMAXVOL_G'], weather_year+1982,
                               bodies={'WTRRSVAR_S' : hydro_WTRRSVAR_S.to_string(),
                                       'WTRRRVAR_T' : hydro_WTRRRVAR_T.to_string(),
                                       'WTRRRFLH' : hydro_WTRRRFLH,
                                       'WTRRSFLH' : hydro_WTRRSFLH,
                                       'HYRSMAXVOL_G' : hydro_HYRSMAXVOL_G})
    
    # Save
    for key in incfiles.keys():
        incfiles[key].save()
        
    with open('Balmorel/base/data/GKFX.inc', 'w') as f:
        f.write(GKFX_start)
        f.write(hydro_GKFX)
        f.write(GKFX_end)


### Main
if __name__ == '__main__':
    CLI()