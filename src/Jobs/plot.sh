###!/bin/sh

scale=$1

if [ $scale = "large" ]; then
  # PtX demands
  python Workflow/Functions/check_ptx_demands.py largescale

  # Capacities
  cd Balmorel
  python analysis/analyse.py --overwrite cap --filters "Scenario in ['noh_eu_rorfix_operun_Iter0','noh2_eu_rorfix_operun_Iter0','h2_eu_rorfix_operun_Iter0','h2_lss_eu_rorfix_operun_Iter0','h2_lss_h2t_eu_rorfix_operun_Iter0']"
  cd ../

  # Seasonal PtX profile
  python Workflow/Analysis.py plot-system-ptx-profile

  # Boxplot
  python Workflow/Functions/boxplot.py model-error-boxplot-largescale

elif [ $scale = "small" ]; then
  # Capacities
  cd Balmorel
  python analysis/analyse.py --overwrite cap --filters "Scenario in ['noh_dispatch_WY2000_Iter0','noh2_dispatch_WY2000_Iter0','h2_dispatch_WY2000_Iter0','h2_lss_dispatch_WY2000_Iter0','h2_lss_h2t_dispatch_WY2000_Iter0']"
  cd ../

  weather_year=1982
  while [ $weather_year -le 2016 ]; do
    # Seasonal PtX profile
    python Workflow/Analysis.py plot-system-ptx-profile $scale --weather-year $weather_year

    # Increment weather year
    weather_year=$((weather_year + 1))
  done

  # Boxplot
  python Workflow/Functions/boxplot.py model-error-boxplot

fi

for scenario in noh noh2 h2 h2_lss h2_lss_h2t; do

  if [ $scenario = "noh" ]; then
    # h2_clustering_technique="vre_availability"
    clustering_name="h2vrehexo"
  else
    # h2_clustering_technique="exogenous_demand"
    clustering_name="h2exohexo"
  fi

  if [ $scale = "large" ]; then
    python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_eu_rorfix_wy2000_1344_${clustering_name} --mc-year 00019 ${scenario}_eu_rorfix_operun_Iter0 flow balmorel
    python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_eu_rorfix_wy2000_1344_${clustering_name} --mc-year 00019 ${scenario}_eu_rorfix_operun_Iter0 flow antares

    # Electricity generation
    python Workflow/Analysis.py plot-all --overwrite ${scenario}_eu_rorfix_operun --mc-year 00019

  elif [ $scale = "small" ]; then
    # Maps
    python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_wy2000_cl1344_${clustering_name} --mc-year 00019 ${scenario}_dispatch_WY2000_Iter0 flow balmorel
    python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_wy2000_cl1344_${clustering_name} --mc-year 00019 ${scenario}_dispatch_WY2000_Iter0 flow antares

    # Electricity generation
    python Workflow/Analysis.py plot-all --overwrite ${scenario}_dispatch_WY2000 --mc-year 00019

  fi
done
