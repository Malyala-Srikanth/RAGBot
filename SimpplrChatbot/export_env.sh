#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    exit 1
fi

# Read .env file line by line and export variables
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" == \#* ]]; then
        continue
    fi

    # Export the variable
    export "$line"
    echo "Exported: $line"
done < .env

echo "All variables from .env file have been exported."
