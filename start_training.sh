#!/bin/bash
set -e

# Activate virtual environment
source env/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# Verify data exists
./scripts/organize_data.sh

# Start training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py 2>&1 | tee logs/training.log 