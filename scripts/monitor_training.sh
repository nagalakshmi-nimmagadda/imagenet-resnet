#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to display GPU stats
show_gpu_stats() {
    echo -e "${BLUE}GPU Statistics:${NC}"
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits
}

# Function to show latest metrics
show_latest_metrics() {
    echo -e "${GREEN}Training Metrics:${NC}"
    python scripts/view_logs.py
}

# Main monitoring loop
while true; do
    clear
    date
    echo "===================="
    
    # Show GPU stats
    show_gpu_stats
    echo "===================="
    
    # Show training metrics
    show_latest_metrics
    
    # Wait before next update
    sleep 30
done 