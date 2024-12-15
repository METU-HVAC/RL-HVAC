#!/bin/bash

# Define the paths to your custom files on the host
EPJSON_PATH="./environment-a403/A403.epJSON"
DEFAULT_JSON_PATH="./environment-a403/A403.json"
DISCRETIZATION_PATH="./environment-a403/custom-constants.py"

# Define the paths inside the container where Sinergym files need to be replaced
SINERGYM_PATH="/usr/local/lib/python3.12/site-packages/sinergym"

# Replace the epJSON file
cp $EPJSON_PATH $SINERGYM_PATH/data/buildings/

# Replace the default JSON file
cp $DEFAULT_JSON_PATH $SINERGYM_PATH/data/default_configuration/

# Add your custom discretization process to constants.py
cp $DISCRETIZATION_PATH $SINERGYM_PATH/utils/constants.py

echo "Files replaced successfully."




