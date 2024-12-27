#!/bin/bash
set -e

# Source config
source utils/config.sh

# Check if Kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Kaggle credentials not found. Please set them up first."
    exit 1
fi

echo "Starting ImageNet download process..."

# Create temporary directory
mkdir -p temp_data

# Download from Kaggle
echo "Downloading ImageNet-1K from Kaggle..."
kaggle datasets download -d ifigotin/imagenetmini-1000 -p temp_data

# Extract data
echo "Extracting dataset..."
unzip temp_data/imagenetmini-1000.zip -d temp_data/

# Organize data
echo "Organizing data structure..."
mv temp_data/imagenet-mini/train/* data/ILSVRC/Data/CLS-LOC/train/
mv temp_data/imagenet-mini/val/* data/ILSVRC/Data/CLS-LOC/val/

# Clean up
rm -rf temp_data

# Upload to S3
echo "Uploading to S3..."
./scripts/sync_s3.sh upload

echo "Dataset download and organization completed!" 