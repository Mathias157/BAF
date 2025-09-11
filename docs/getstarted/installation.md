# Installation

[GAMS 47+](https://www.gams.com/download/), [Antares 8.7](https://github.com/AntaresSimulatorTeam/Antares_Simulator/releases/tag/v8.7.0) and the described [Python environment](#python-environment) are required to run the BAF. Running pre-processing (and some of the visualisations) also requires [R 4.2.3](https://cran.r-project.org/) and the setup described in [R setup](#r-setup). 

Additionally, data is required which is described in the next page.

## Python Environment
Can be installed using [pixi](https://pixi.sh/latest/). A pixi installation will ensure that *all* packages are the same as in the time of developing BAF.

:::{warning}
Remember to have R in your [PATH environment variable](https://superuser.com/questions/284342/what-are-path-and-other-environment-variables-and-how-can-i-set-or-use-them) before the environment installation below! At least if you wish to use the mentioned R functionalities.
:::

Simply [install pixi](https://pixi.sh/latest/#installation), cd into this repository and run `pixi install`. 

## R Setup
The pre-processing scripts and some visualisations use [R packages for Antares](https://github.com/rte-antares-rpackage), developed by RTE. Open up R and install them with the commands:
```
install.packages("antaresViz")
```