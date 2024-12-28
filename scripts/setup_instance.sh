#!/bin/bash
set -e

echo "Setting up training instance..."

# Function to check if reboot is needed
needs_reboot() {
    if [ -f /var/run/reboot-required ]; then
        return 0
    fi
    return 1
}

# Function to optimize system settings
optimize_system() {
    # Disable unnecessary services
    sudo systemctl stop snapd.service
    sudo systemctl disable snapd.service
    sudo systemctl stop snapd.socket
    sudo systemctl disable snapd.socket
    
    # Set CPU governor to performance
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    
    # Optimize memory settings
    echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
    echo "vm.vfs_cache_pressure=50" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
    
    # Optimize network settings
    echo "net.core.rmem_max=2147483647" | sudo tee -a /etc/sysctl.conf
    echo "net.core.wmem_max=2147483647" | sudo tee -a /etc/sysctl.conf
    echo "net.ipv4.tcp_rmem=4096 87380 2147483647" | sudo tee -a /etc/sysctl.conf
    echo "net.ipv4.tcp_wmem=4096 65536 2147483647" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
}

# Remove existing NVIDIA installations
echo "Removing existing NVIDIA installations..."
sudo apt-get remove --purge -y '^nvidia-.*' '^cuda-.*' '^libnvidia-.*' || true
sudo apt-get autoremove -y
sudo apt-get clean

# Add GPU drivers repository
echo "Adding GPU drivers repository..."
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update

# Install only necessary packages
echo "Installing prerequisites..."
sudo apt-get install -y --no-install-recommends \
    linux-headers-$(uname -r) \
    build-essential \
    dkms \
    gcc \
    make \
    python3-pip \
    python3-dev \
    python3-venv \
    nvidia-driver-535

# Install CUDA toolkit
echo "Installing CUDA toolkit..."
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
chmod +x cuda_12.2.0_535.54.03_linux.run
sudo ./cuda_12.2.0_535.54.03_linux.run --silent --toolkit --no-opengl-libs --no-man-page --override
cd -

# Set up environment variables
echo "Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF
source ~/.bashrc

# Create and activate virtual environment
echo "Setting up Python environment..."
python3 -m venv env
source env/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -r requirements.txt

# Optimize NVIDIA settings
echo "Optimizing GPU settings..."
sudo nvidia-smi -pm 1
sudo nvidia-smi --auto-boost-default=0
sudo nvidia-smi -pl 250  # Adjust power limit for efficiency

# Run system optimizations
optimize_system

# Clean up
rm -f /tmp/cuda_12.2.0_535.54.03_linux.run
sudo apt-get clean
sudo apt-get autoremove -y

# Create cache directories
mkdir -p ~/.cache/torch/hub/checkpoints
mkdir -p ~/.cache/torch/hub/pytorch_vision_v0.10.0

# Print status and next steps
echo "Installation completed!"
if needs_reboot; then
    echo "System requires a reboot. Please run: sudo reboot"
else
    echo "Running final verification..."
    nvidia-smi
    python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
fi

echo "
Cost optimization features enabled:
1. Minimal package installation
2. System optimization for ML workloads
3. GPU power optimization
4. Memory management settings
5. Network optimization
6. Disk I/O optimization
7. Service optimization
8. Cache preparation
" 