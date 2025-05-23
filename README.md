# Balmorel and Antares Soft-Linking Framework

This Balmorel-Antares soft-linking framework (BAF) was used in [Rosendal et al. 2025](https://doi.org/10.1016/j.apenergy.2025.125512), in an investigation of soft-linking strategies for coupling investment and operational models. The specific data of that study could unfortunately not be shared, but the framework will be reused here for a second, open-data application, using the [OSMOSE WP1 dataset](https://zenodo.org/records/7323821). 

Get started by reading the [documentation](https://github.com/Mathias157/BAF/blob/master/docs/Balmorel_Antares_Soft_Coupling_Framework_Documentation.pdf).

## Installation

[GAMS 47+](https://www.gams.com/download/), [Antares 8.7](https://github.com/AntaresSimulatorTeam/Antares_Simulator/releases/tag/v8.7.0) and the described [Python environment](#python-environment) are required to run the BAF. Running pre-processing (and some of the visualisations) also requires [R 4.2.3](https://cran.r-project.org/) and the setup described in [R setup](#r-setup). 

### Python Environment
Can be installed using [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [pixi](https://pixi.sh/latest/). A pixi installation will ensure that *all* packages are the same as in the time of developing BAF, while a conda installation through the `environment.yaml` could lead to different sub-packages being installed.

For pixi, simply [install pixi](https://pixi.sh/latest/#installation) and run `pixi install` in the top level of the folder. For conda, run `conda env create -f environment.yaml`.

### R Setup
The pre-processing scripts and some visualisations use [R packages for Antares](https://github.com/rte-antares-rpackage), developed by RTE. Open up R and install them with the commands:
```
install.packages("antaresViz")
```

## Storing the Data
The Antares and Balmorel *frameworks* are stored in git, as well as input data for Balmorel. However, input data for Antares is not tracked, and neither is important static files in the Pre-Processing folder. Download them [here](https://data.dtu.dk) and extract into the src folder. To store adjustments to these static files, execute the following commands in powershell to zip data:

Windows:
```
powershell Compress-Archive -Path "Antares/input, Pre-Processing/Data, Pre-Processing/Output" -DestinationPath "BAF-Data_branch_version.zip"
```

Linux:
```
zip -r -q BAF-Data_branch_version.zip Pre-Processing/Output Pre-Processing/Data Antares/input
```

## Unzipping the Data on HPC
Use the following command to unzip - -qq disables logging output, -o overwrites existing files 
unzip -qq -o BAF-Data_branch_version.zip

If unzipping the data file on a HPC, you may need to ensure writing capabilities on the extracted files by doing the following commands on the extracted folders: 
```
chmod -R +x data
chmod -R +x Pre-Processing
chmod -R +x input
```

Otherwise, these files will not be editable, which is needed in the framework