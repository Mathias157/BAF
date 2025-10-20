###!/bin/sh

scale=$1

for scenario in noh noh2 h2; do

  if [ $scenario = "noh" ]; then
    # h2_clustering_technique="vre_availability"
    clustering_name="h2vrehexo"
  else
    # h2_clustering_technique="exogenous_demand"
    clustering_name="h2exohexo"
  fi

  if [ $scale = "large" ]; then

    # Maps
    python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_${clustering_name}_stofixflowbased --mc-year 00019 ${scenario}_eu_operun_flowbased_Iter0 flow balmorel
    python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_${clustering_name}_stofixflowbased --mc-year 00019 ${scenario}_eu_operun_flowbased_Iter0 flow antares

    # Electricity generation
    python Workflow/Analysis.py plot-all --overwrite ${scenario}_eu_operun_flowbased --mc-year 00019

    # Seasonal PtX profile
    python Workflow/Analysis.py plot-system-ptx-profile $scenario large
  elif [ $scale = "small" ]; then
    # Maps
    python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_wy2000_cl1344_${clustering_name} --mc-year 00019 ${scenario}_dispatch_WY2000_Iter0 flow balmorel
    python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_wy2000_cl1344_${clustering_name} --mc-year 00019 ${scenario}_dispatch_WY2000_Iter0 flow antares

    # Electricity generation
    python Workflow/Analysis.py plot-all --overwrite ${scenario}_dispatch_WY2000 --mc-year 00019

  fi
done

if [ $scale = "large" ]; then
  # Capacities
  cd Balmorel
  python analysis/analyse.py --overwrite cap --filters "Scenario in ['noh_eu_operun_flowbased_Iter0','noh2_eu_operun_flowbased_Iter0','h2_eu_operun_flowbased_Iter0','h2_lss_eu_operun_flowbased_Iter0','h2_lss_h2t_eu_operun_flowbased_Iter0']"
  cd ../

  # Seasonal PtX profile
  python Workflow/Analysis.py plot-system-ptx-profile $scale

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
