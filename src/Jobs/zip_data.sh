#!/bin/bash

# Store the short hash in a variable
hash=$(git rev-parse --short=8 HEAD)

# Use the variable
zip -r -q BAF-Data_EU-system_$hash.zip Pre-Processing/Output Pre-Processing/Data Antares/input
