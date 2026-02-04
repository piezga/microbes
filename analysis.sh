#!/bin/bash

echo "=== Starting analysis pipeline ==="

# Run first script
echo "Fitting the model and getting p-values..."
python -m scripts.get_p_values

# Check if it succeeded
if [ $? -ne 0 ]; then
    echo "Script 1 failed! Exiting."
    exit 1
fi

# Run second script  
echo "Projecting onto microbes layer..."
python -m scripts.projection

if [ $? -ne 0 ]; then
    echo "Script 2 failed! Exiting."
    exit 1
fi

echo "=== All scripts completed successfully ==="
