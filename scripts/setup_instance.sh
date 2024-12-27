#!/bin/bash

# Exit on error
set -e

# Source config
source utils/config.sh

echo "Setting up ImageNet training environment..."

# Install system packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv unzip

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install requirements
pip install -r requirements.txt

# Configure AWS region
aws configure set region $aws_region

# Create necessary directories
mkdir -p data/ILSVRC/Data/CLS-LOC/{train,val}
mkdir -p checkpoints logs

# Make scripts executable
chmod +x scripts/*.sh utils/*.sh
chmod +x start_training.sh

echo "Setup completed successfully!" 