#!/bin/bash
# Convenience script to run Bitcoin classification experiments
# Usage: ./run_experiment.sh [config_name] [experiment_type]

set -e

# Get the project root directory (go up from scripts/bitcoin/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Default config and experiment type
CONFIG_NAME=${1:-"simple"}
EXPERIMENT_TYPE=${2:-"standard"}

# Map config names to files
case $CONFIG_NAME in
    "simple")
        CONFIG_FILE="scripts/bitcoin/btc_reduced_simple_config.yaml"
        ;;
    "full")
        CONFIG_FILE="scripts/bitcoin/btc_reduced_classification_config.yaml"
        ;;
    "unified")
        CONFIG_FILE="scripts/bitcoin/btc_reduced_unified_config.yaml"
        ;;
    "balanced")
        CONFIG_FILE="scripts/bitcoin/btc_reduced_balanced_config.yaml"
        EXPERIMENT_TYPE="balanced"
        ;;
    *)
        CONFIG_FILE="scripts/bitcoin/$CONFIG_NAME"
        ;;
esac

# Choose experiment script
case $EXPERIMENT_TYPE in
    "balanced")
        SCRIPT="scripts/bitcoin/classification_experiment_balanced.py"
        ;;
    "standard"|*)
        SCRIPT="scripts/bitcoin/classification_experiment.py"
        ;;
esac

echo "Running Bitcoin Classification Experiment"
echo "========================================="
echo "Config: $CONFIG_FILE"
echo "Script: $SCRIPT"
echo "Working directory: $PROJECT_ROOT"
echo ""

# Check if files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available configs:"
    ls -1 scripts/bitcoin/*.yaml
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Experiment script not found: $SCRIPT"
    exit 1
fi

# Run the experiment
python "$SCRIPT" "$CONFIG_FILE"

echo ""
echo "Experiment completed!"
echo "Results saved in the output directory specified in the config file."