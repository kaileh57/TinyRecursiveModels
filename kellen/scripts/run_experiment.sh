#!/bin/bash
# Run a single experiment on TPU v4-64 (stable-1 node)

set -e

# Parse arguments
EXPERIMENT_CONFIG=$1
RUN_NAME=$2

if [ -z "$EXPERIMENT_CONFIG" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 <experiment_config> <run_name>"
    echo "Example: $0 e1_1_baseline baseline_run_1"
    exit 1
fi

CONFIG_PATH="kellen/configs/experiments/${EXPERIMENT_CONFIG}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "=== Running Experiment ==="
echo "Config: $CONFIG_PATH"
echo "Run name: $RUN_NAME"
echo "TPU: v4-64 (8 workers)"
echo

# Set XLA environment variables for TPU v4
export XLA_USE_BF16=1
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000
export PJRT_DEVICE=TPU

# Launch training on all 8 workers
python3 kellen/src/train_tpu.py \
    --config=$CONFIG_PATH \
    --run_name=$RUN_NAME \
    2>&1 | tee logs/${RUN_NAME}.log

echo "Experiment complete!"
