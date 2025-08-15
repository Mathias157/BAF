# Installation

[GAMS 47+](https://www.gams.com/download/), [Antares 8.7](https://github.com/AntaresSimulatorTeam/Antares_Simulator/releases/tag/v8.7.0) and the described [Python environment](#python-environment) are required to run the BAF. Running pre-processing (and some of the visualisations) also requires [R 4.2.3](https://cran.r-project.org/) and the setup described in [R setup](#r-setup). 

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

## Data

The framework depends on data for Balmorel, Antares and other raw data if running pre-processing scripts is desired


### Balmorel Data

The data for Balmorel is stored in the [following repository.](https://github.com/Mathias157/Balmorel_data/tree/BAF_small-system)
Follow these commands to download and configure it to Balmorel's expectations:

```bash
cd src/Balmorel/base
git clone https://github.com/Mathias157/Balmorel_data.git
mv Balmorel_data data
cd data
git switch BAF_small-system
```

### Antares and Raw Data

Input data for Antares is not tracked, and neither are important static files in the Pre-Processing folder. This will become available at [data.dtu.dk](https://data.dtu.dk) at some stage, and is available upon request.

Extract the contents of the .zip file into the src folder.
The following command can be used to unzip in Linux (-qq disables logging output, -o overwrites existing files):
``` 
unzip -qq -o BAF-Data_branch_version.zip
```

If unzipping the data file on a HPC, you may need to ensure writing capabilities on the extracted files by doing the following commands on the extracted folders: 
```
chmod -R +x Pre-Processing/Output
chmod -R +x Pre-Processing/Data
chmod -R +x input
```

Otherwise, these files will not be editable, which is needed in the framework

