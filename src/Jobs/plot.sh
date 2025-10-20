###!/bin/sh
for scenario in noh noh2 h2; do

  if [ $scenario = "noh" ]; then
    # h2_clustering_technique="vre_availability"
    clustering_name="h2vrehexo"
  else
    # h2_clustering_technique="exogenous_demand"
    clustering_name="h2exohexo"
  fi

  # Maps
  python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_${clustering_name}_stofixflowbased --mc-year 00019 ${scenario}_eu_operun_flowbased_Iter0 flow balmorel
  python Workflow/Functions/plot_map.py --scenario-folder $scenario --antares-scenario ${scenario}_${clustering_name}_stofixflowbased --mc-year 00019 ${scenario}_eu_operun_flowbased_Iter0 flow antares

  # Electricity generation
  python Workflow/Analysis.py plot-all --overwrite ${scenario}_eu_operun_flowbased --mc-year 00019

  # Boxplot
  python Workflow/Functions/boxplot.py model-error-boxplot-largescale

  # Seasonal PtX profile
  python Workflow/Analysis.py plot-system-ptx-profile $scenario large

done
