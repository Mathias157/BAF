# python Workflow/Analysis.py plot noh2_fullyear
# python Workflow/Analysis.py plot h2_fullyear
# python Workflow/Analysis.py plot h2_lss_fullyear

python Workflow/Functions/plot_map.py --scenario-folder noh --antares-scenario noh_wy2000_cl1344 --mc-year 00019 noh_dispatch_WY2000_Iter0 flow balmorel
python Workflow/Functions/plot_map.py --scenario-folder noh --antares-scenario noh_wy2000_cl1344 --mc-year 00019 noh_dispatch_WY2000_Iter0 flow antares

python Workflow/Functions/plot_map.py --scenario-folder noh2 --antares-scenario noh2_wy2000_cl1344 --mc-year 00019 noh2_dispatch_WY2000_Iter0 flow balmorel
python Workflow/Functions/plot_map.py --scenario-folder noh2 --antares-scenario noh2_wy2000_cl1344 --mc-year 00019 noh2_dispatch_WY2000_Iter0 flow antares

python Workflow/Functions/plot_map.py --scenario-folder h2 --antares-scenario h2_wy2000_cl1344 --mc-year 00019 h2_dispatch_WY2000_Iter0 flow balmorel
python Workflow/Functions/plot_map.py --scenario-folder h2 --antares-scenario h2_wy2000_cl1344 --mc-year 00019 h2_dispatch_WY2000_Iter0 flow antares

python Workflow/Functions/plot_map.py --scenario-folder h2_lss --antares-scenario h2_lss_wy2000_cl1344 --mc-year 00019 h2_lss_dispatch_WY2000_Iter0 flow balmorel
python Workflow/Functions/plot_map.py --scenario-folder h2_lss --antares-scenario h2_lss_wy2000_cl1344 --mc-year 00019 h2_lss_dispatch_WY2000_Iter0 flow antares

python Workflow/Functions/plot_map.py --scenario-folder h2_lss_h2t --antares-scenario h2_lss_h2t_wy2000_cl1344 --mc-year 00019 h2_lss_h2t_dispatch_WY2000_Iter0 flow balmorel
python Workflow/Functions/plot_map.py --scenario-folder h2_lss_h2t --antares-scenario h2_lss_h2t_wy2000_cl1344 --mc-year 00019 h2_lss_h2t_dispatch_WY2000_Iter0 flow antares
