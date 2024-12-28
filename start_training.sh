#!/bin/bash
set -e

# Get the actual network interface name
NETWORK_INTERFACE=$(ip route get 8.8.8.8 | awk '{print $5; exit}')
echo "Using network interface: $NETWORK_INTERFACE"

# Basic environment variables
export CUDA_LAUNCH_BLOCKING=0
export PYTHONFAULTHANDLER=1

# NCCL Configuration
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH  # Only show initialization and graph creation
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# NCCL Performance Tuning
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=2097152
export NCCL_NET_GDR_LEVEL=0
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=4

# CUDA Optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# CPU Threading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Create required directories
mkdir -p checkpoints logs

# Create timestamped log file
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# Show GPU information
echo "GPU Information:" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE"

# Optimize GPU settings for NVIDIA L4
echo "Optimizing GPU settings for NVIDIA L4..." | tee -a "$LOG_FILE"

# Check if running as root, if not, use sudo
if [ "$EUID" -ne 0 ]; then
    echo "Need sudo rights for GPU optimization..."
    # Enable persistence mode
    sudo nvidia-smi -pm 1
    
    # Set power limits and compute mode for each GPU
    for i in {0..3}; do
        echo "Configuring GPU $i..." | tee -a "$LOG_FILE"
        sudo nvidia-smi -i $i -pl 70
        sudo nvidia-smi -i $i --compute-mode=0
        sudo nvidia-smi -i $i --applications-clocks=1305,7000
    done
else
    nvidia-smi -pm 1
    for i in {0..3}; do
        nvidia-smi -i $i -pl 70
        nvidia-smi -i $i --compute-mode=0
        nvidia-smi -i $i --applications-clocks=1305,7000
    done
fi

# Verify network interface
echo "Network Interface Configuration:" | tee -a "$LOG_FILE"
ip addr show $NETWORK_INTERFACE | tee -a "$LOG_FILE"

# DDP Configuration
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
export WORLD_SIZE=4
export LOCAL_RANK="$SLURM_LOCALID"
export RANK="$SLURM_PROCID"

# Print configuration
echo "Training Configuration:" | tee -a "$LOG_FILE"
echo "NCCL_SOCKET_IFNAME: $NETWORK_INTERFACE" | tee -a "$LOG_FILE"
echo "MASTER_ADDR: $MASTER_ADDR" | tee -a "$LOG_FILE"
echo "WORLD_SIZE: $WORLD_SIZE" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"

# Start training with distributed launch
echo "Starting training..." | tee -a "$LOG_FILE"

torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    --node_rank=0 \
    --nnodes=1 \
    train.py 2>&1 | tee -a "$LOG_FILE"

# Create a symlink to latest log
ln -sf "$LOG_FILE" logs/latest.log 

# Add after starting training
echo "To monitor training, in another terminal run:"
echo "./scripts/monitor_training.sh" 