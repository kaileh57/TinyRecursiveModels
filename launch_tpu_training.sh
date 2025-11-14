#!/bin/bash
#
# Multi-host launcher for TPU v4-64 (8 workers)
#
# This script launches training on all 8 TPU workers simultaneously,
# with proper JAX distributed setup for multi-host training.
#
# Usage:
#   ./launch_tpu_training.sh <experiment_name> [tpu_name] [zone]
#
# Example:
#   ./launch_tpu_training.sh baseline
#   ./launch_tpu_training.sh exp04a stable-1 us-central2-b
#

set -euo pipefail

# Configuration
EXPERIMENT="${1:-baseline}"
TPU_NAME="${2:-${TPU_NAME:-stable-1}}"
ZONE="${3:-${ZONE:-us-central2-b}}"
NUM_WORKERS=8
PROJECT_DIR="${PROJECT_DIR:-~/TinyRecursiveModels}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================"
echo "TPU v4-64 Multi-Host Training Launcher"
echo "================================================================"
echo "Experiment:     ${EXPERIMENT}"
echo "TPU Name:       ${TPU_NAME}"
echo "Zone:           ${ZONE}"
echo "Workers:        ${NUM_WORKERS}"
echo "Project Dir:    ${PROJECT_DIR}"
echo "================================================================"

# Create log directory
LOG_DIR="kellen/logs/tpu_workers"
mkdir -p "${LOG_DIR}"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG_DIR="${LOG_DIR}/${EXPERIMENT}_${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

echo -e "${GREEN}Created log directory: ${RUN_LOG_DIR}${NC}"

# Function to launch on a single worker
launch_worker() {
    local worker_id=$1
    local log_file="${RUN_LOG_DIR}/worker_${worker_id}.log"

    echo -e "${YELLOW}Launching worker ${worker_id}...${NC}"

    # Build the command to run on the TPU worker
    # We need to:
    # 1. Set JAX environment variables
    # 2. Change to project directory
    # 3. Run the training script

    local remote_cmd="
        set -e
        export JAX_PROCESS_COUNT=${NUM_WORKERS}
        export JAX_PROCESS_INDEX=${worker_id}
        export JAX_PLATFORMS=tpu
        export TF_CPP_MIN_LOG_LEVEL=0

        cd ${PROJECT_DIR}

        echo '=== Worker ${worker_id} Environment ==='
        echo \"JAX_PROCESS_COUNT=\${JAX_PROCESS_COUNT}\"
        echo \"JAX_PROCESS_INDEX=\${JAX_PROCESS_INDEX}\"
        echo \"PWD=\$(pwd)\"
        echo '======================================'

        # Run the experiment
        python pretrain_jax.py --config-name=${EXPERIMENT}
    "

    # Launch via gcloud ssh (runs in background)
    gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
        --zone="${ZONE}" \
        --worker="${worker_id}" \
        --command="${remote_cmd}" \
        > "${log_file}" 2>&1 &

    # Save the PID
    local pid=$!
    echo "${pid}" > "${RUN_LOG_DIR}/worker_${worker_id}.pid"

    echo -e "${GREEN}Worker ${worker_id} launched (PID: ${pid}, Log: ${log_file})${NC}"
}

# Function to check worker status
check_workers() {
    local all_running=true

    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local pid_file="${RUN_LOG_DIR}/worker_${i}.pid"
        if [[ -f "${pid_file}" ]]; then
            local pid=$(cat "${pid_file}")
            if ps -p "${pid}" > /dev/null 2>&1; then
                echo -e "${GREEN}Worker ${i}: Running (PID: ${pid})${NC}"
            else
                echo -e "${RED}Worker ${i}: Stopped (PID: ${pid})${NC}"
                all_running=false
            fi
        else
            echo -e "${RED}Worker ${i}: Not launched${NC}"
            all_running=false
        fi
    done

    if [[ "${all_running}" == "true" ]]; then
        echo -e "\n${GREEN}All workers are running${NC}"
    else
        echo -e "\n${YELLOW}Some workers have stopped${NC}"
    fi
}

# Function to tail logs
tail_logs() {
    local worker_id="${1:-0}"
    local log_file="${RUN_LOG_DIR}/worker_${worker_id}.log"

    if [[ -f "${log_file}" ]]; then
        echo -e "${GREEN}Tailing log for worker ${worker_id}:${NC}"
        tail -f "${log_file}"
    else
        echo -e "${RED}Log file not found: ${log_file}${NC}"
    fi
}

# Launch all workers
echo ""
echo "================================================================"
echo "Launching ${NUM_WORKERS} workers..."
echo "================================================================"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    launch_worker ${i}
    # Small delay between launches to avoid overwhelming SSH
    sleep 2
done

echo ""
echo "================================================================"
echo "All workers launched!"
echo "================================================================"
echo ""
echo "Monitor progress with:"
echo "  - Tail worker 0 log:  tail -f ${RUN_LOG_DIR}/worker_0.log"
echo "  - Check all workers:  tail -f ${RUN_LOG_DIR}/worker_*.log"
echo "  - Worker status:      ps aux | grep pretrain_jax"
echo ""
echo "Log directory: ${RUN_LOG_DIR}"
echo ""
echo -e "${YELLOW}Waiting for all workers to complete...${NC}"
echo ""

# Wait for all background jobs to complete
wait

echo ""
echo "================================================================"
echo "All workers completed!"
echo "================================================================"
echo ""
echo "Check logs in: ${RUN_LOG_DIR}"
echo ""

# Check final status
check_workers

# Check if any worker failed by looking at exit codes in logs
echo ""
echo "Checking for errors..."
failed=0
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    log_file="${RUN_LOG_DIR}/worker_${i}.log"
    if grep -q "ERROR" "${log_file}" || grep -q "FAILED" "${log_file}" || grep -q "Traceback" "${log_file}"; then
        echo -e "${RED}Worker ${i}: Found errors in log${NC}"
        failed=$((failed + 1))
    else
        echo -e "${GREEN}Worker ${i}: No errors detected${NC}"
    fi
done

if [[ ${failed} -gt 0 ]]; then
    echo -e "\n${RED}Training completed with ${failed} workers reporting errors${NC}"
    exit 1
else
    echo -e "\n${GREEN}Training completed successfully on all workers${NC}"
    exit 0
fi
