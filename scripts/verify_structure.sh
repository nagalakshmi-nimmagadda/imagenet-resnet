#!/bin/bash
set -e

# Check required directories
for dir in data/ILSVRC/Data/CLS-LOC/{train,val} checkpoints logs; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# Verify CUDA setup
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Optimize system settings
echo "Optimizing system settings..."

# Increase file descriptors
ulimit -n 65535

# Set CPU governor to performance
if [ -w /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor ]; then
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
fi

# Optimize TCP settings
sudo sysctl -w net.core.rmem_max=2147483647
sudo sysctl -w net.core.wmem_max=2147483647
sudo sysctl -w net.ipv4.tcp_rmem='4096 87380 2147483647'
sudo sysctl -w net.ipv4.tcp_wmem='4096 65536 2147483647'

# Set GPU power limits for efficiency
nvidia-smi -pm 1
nvidia-smi --auto-boost-default=0
nvidia-smi -pl 250  # Adjust based on your GPU

echo "Setup verification and optimization complete!" 