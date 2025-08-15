# Data

The framework depends on data for Balmorel, Antares and other raw data if running pre-processing scripts is desired

## Balmorel Data

The data for Balmorel is stored in the [following repository.](https://github.com/Mathias157/Balmorel_data/tree/BAF_small-system)
Follow these commands to download and configure it to Balmorel's expectations:

```bash
cd src/Balmorel/base
git clone https://github.com/Mathias157/Balmorel_data.git
mv Balmorel_data data
cd data
git switch BAF_small-system
```

## Antares and Raw Data

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

