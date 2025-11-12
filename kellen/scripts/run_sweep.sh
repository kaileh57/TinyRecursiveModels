#!/bin/bash
# Run a sweep of experiments sequentially

set -e

SWEEP_NAME=$1
if [ -z "$SWEEP_NAME" ]; then
    echo "Usage: $0 <sweep_name>"
    echo "Available sweeps: tn_sweep, batch_sweep, all_ablations"
    exit 1
fi

case $SWEEP_NAME in
    tn_sweep)
        echo "=== Running T/n Schedule Sweep ==="
        experiments=(
            "e2_2_t2_n2:tn_t2_n2"
            "e2_2_t2_n6:tn_t2_n6"
            "e2_2_t3_n4:tn_t3_n4"
            "e2_2_t3_n6:tn_t3_n6"
            "e2_2_t4_n6:tn_t4_n6"
        )
        ;;
    batch_sweep)
        echo "=== Running Batch Size Sweep ==="
        experiments=(
            "e1_1_baseline:batch_256"
            "e1_3_batch_512:batch_512"
            "e1_3_batch_1024:batch_1024"
        )
        ;;
    all_ablations)
        echo "=== Running All Core Ablations ==="
        experiments=(
            "e1_1_baseline:baseline"
            "e2_1_ema_off:ema_off"
            "e2_2_t3_n6:tn_baseline"
            "e2_3_attention:attention"
            "e2_4_nsup_8:nsup_8"
        )
        ;;
    *)
        echo "Unknown sweep: $SWEEP_NAME"
        exit 1
        ;;
esac

for exp in "${experiments[@]}"; do
    IFS=':' read -r config name <<< "$exp"
    echo ""
    echo "========================================"
    echo "Running: $name (config: $config)"
    echo "========================================"
    ./kellen/scripts/run_experiment.sh $config $name
done

echo ""
echo "=== Sweep Complete ==="
