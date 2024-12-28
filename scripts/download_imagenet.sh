#!/bin/bash
set -e

# Source config
source utils/config.sh

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip unzip aria2 pv
pip install --upgrade kaggle
check_status "Dependencies installation"

# Setup Kaggle credentials
echo "Setting up Kaggle credentials..."
mkdir -p ~/.kaggle
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Please enter your Kaggle credentials:"
    read -p "Username: " username
    read -p "Key: " key
    echo "{\"username\":\"$username\",\"key\":\"$key\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
fi

# Create directories
mkdir -p data/ILSVRC/Data/CLS-LOC/{train,val}
mkdir -p raw_data
check_status "Directory creation"

# Download ImageNet using Kaggle API
echo "Downloading ImageNet data..."
kaggle competitions download -c imagenet-object-localization-challenge -p raw_data
check_status "Data download"

# Extract data with progress
echo "Extracting dataset..."
unzip -q raw_data/imagenet-object-localization-challenge.zip -d raw_data/ | pv -l
check_status "Data extraction"

# Organize data using find
echo "Organizing training data..."
cd raw_data/ILSVRC/Data/CLS-LOC/train
for class_dir in */; do
    echo "Moving class: $class_dir"
    mkdir -p ../../../../data/ILSVRC/Data/CLS-LOC/train/"$class_dir"
    find "$class_dir" -type f -exec mv {} ../../../../data/ILSVRC/Data/CLS-LOC/train/"$class_dir" \;
done
cd ../../../../../

echo "Organizing validation data..."
cd raw_data/ILSVRC/Data/CLS-LOC/val
for class_dir in */; do
    echo "Moving class: $class_dir"
    mkdir -p ../../../../data/ILSVRC/Data/CLS-LOC/val/"$class_dir"
    find "$class_dir" -type f -exec mv {} ../../../../data/ILSVRC/Data/CLS-LOC/val/"$class_dir" \;
done
cd ../../../../../

# Verify data organization
echo "Verifying data organization..."
train_classes=$(ls data/ILSVRC/Data/CLS-LOC/train | wc -l)
val_classes=$(ls data/ILSVRC/Data/CLS-LOC/val | wc -l)
echo "Found $train_classes classes in training set"
echo "Found $val_classes classes in validation set"

if [ "$train_classes" -ne 1000 ] || [ "$val_classes" -ne 1000 ]; then
    echo "Error: Expected 1000 classes, but found different numbers"
    exit 1
fi

# Clean up
echo "Cleaning up..."
rm -rf raw_data
check_status "Cleanup"

# Upload to S3 with multipart and resume
echo "Creating S3 bucket..."
aws s3 mb s3://${aws_bucket_name} || true

echo "Uploading to S3..."
aws s3 sync data/ILSVRC/Data/CLS-LOC/ s3://${aws_bucket_name}/ILSVRC/Data/CLS-LOC/ \
    --progress \
    --only-show-errors \
    --storage-class STANDARD \
    --multipart-threshold 64MB \
    --multipart-chunksize 64MB
check_status "S3 upload"

echo "ImageNet download and processing completed successfully!" 