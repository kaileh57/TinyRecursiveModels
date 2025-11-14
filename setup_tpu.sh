#!/bin/bash
# TPU v4-64 Setup Script for JAX Training
# This script sets up the environment for training on TPU v4-64 (32 chips, 64 cores, 8x4 workers)

set -e

echo "========================================="
echo "TPU v4-64 Setup for JAX Training"
echo "========================================="

# Install JAX with TPU support
echo "Installing JAX with TPU support..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify JAX installation
echo "Verifying JAX installation..."
python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}'); print(f'Device count: {jax.device_count()}')"

# TPU-specific environment variables
echo "Setting TPU environment variables..."
export TPU_NUM_DEVICES=64
export TPU_CHIPS_PER_HOST_BOUNDS=8,4
export JAX_PLATFORMS=tpu

# Verify TPU setup
echo "Verifying TPU configuration..."
python3 -c "
import jax
import jax.numpy as jnp

print('='*50)
print('TPU Configuration:')
print('='*50)
print(f'JAX version: {jax.__version__}')
print(f'Number of devices: {jax.device_count()}')
print(f'Number of processes: {jax.process_count()}')
print(f'Process index: {jax.process_index()}')
print(f'Local device count: {jax.local_device_count()}')
print(f'Devices: {jax.devices()}')
print('='*50)

# Test basic TPU computation
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print(f'Test computation successful! Result shape: {y.shape}')
print('='*50)
"

echo "========================================="
echo "TPU v4-64 setup complete!"
echo "Ready to start training with JAX"
echo "========================================="
echo ""
echo "To start training, run:"
echo "  python pretrain_jax.py"
echo ""
echo "For distributed training across all TPU cores:"
echo "  python -m jax.distributed.initialize --coordinator_address=<coordinator_ip>:1234 --num_processes=<num_hosts> --process_id=<host_id> pretrain_jax.py"
