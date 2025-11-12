#!/bin/bash
# Convenience script for running TRM experiments
# Usage: ./kellen/run.sh EXPERIMENT_NAME

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "TRM Scaling Research - Experiment Runner"
echo "========================================="
echo ""

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 EXPERIMENT_NAME [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  $0 baseline                    # Run baseline experiment"
    echo "  $0 exp01a                      # Run experiment 01a"
    echo "  $0 baseline --dry-run          # Dry run (test config)"
    echo "  $0 --list                      # List all experiments"
    echo "  $0 --help                      # Show this help"
    echo ""
    exit 1
fi

# Handle special flags
if [ "$1" == "--list" ]; then
    python "$SCRIPT_DIR/experiments/run_experiment_batch.py" --list
    exit 0
fi

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "TRM Experiment Runner"
    echo ""
    echo "Usage: $0 EXPERIMENT_NAME [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run          Test configuration without training"
    echo "  --list             List all available experiments"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Environment:"
    echo "  TPU_NAME          TPU node name (default: stable-1)"
    echo "  NUM_WORKERS       Number of workers (default: 8)"
    echo ""
    echo "Examples:"
    echo "  $0 baseline                           # Run baseline"
    echo "  $0 exp01a                             # Run exp01a"
    echo "  TPU_NAME=my-tpu $0 baseline           # Custom TPU"
    echo "  $0 baseline --dry-run                 # Test config"
    echo ""
    exit 0
fi

# Get TPU name from environment or use default
TPU_NAME="${TPU_NAME:-stable-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

EXPERIMENT=$1
shift  # Remove experiment name from arguments

echo "Experiment: $EXPERIMENT"
echo "TPU: $TPU_NAME"
echo "Workers: $NUM_WORKERS"
echo ""

# Run the experiment
python "$SCRIPT_DIR/experiments/run_experiment.py" \
    "$EXPERIMENT" \
    --tpu-name "$TPU_NAME" \
    --num-workers "$NUM_WORKERS" \
    "$@"
