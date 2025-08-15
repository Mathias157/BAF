# Development

Some guides on developing BAF.

## Storing Data for Antares and Pre-Processing
To store adjustments to the format of the mentioned .zip-file, execute the following commands in powershell to zip data:

Windows:
```
powershell Compress-Archive -Path "Antares/input, Pre-Processing/Data, Pre-Processing/Output" -DestinationPath "BAF-Data_branch_version.zip"
```

Linux:
```
zip -r -q BAF-Data_branch_version.zip Pre-Processing/Output Pre-Processing/Data Antares/input
```