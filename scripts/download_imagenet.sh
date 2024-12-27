#!/bin/bash
set -e

# Source config for bucket name
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
sudo apt-get install -y transmission-cli python3-pip
pip install academictorrents-python tqdm
check_status "Dependencies installation"

# Create directories
mkdir -p raw_data
mkdir -p data/ILSVRC/Data/CLS-LOC/{train,val}
check_status "Directory creation"

echo "Downloading ImageNet data..."
python3 - <<'EOF'
from download_utils import download_with_progress

# Download training data
train_path = download_with_progress(
    'a306397ccf9c2ead27155983c254227c0fd938e2',
    'raw_data',
    'Downloading training data'
)

# Download validation data
val_path = download_with_progress(
    '5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5',
    'raw_data',
    'Downloading validation data'
)
EOF
check_status "Data download"

echo "Processing training data..."
cd data/ILSVRC/Data/CLS-LOC/train
tar -xf ../../../../../raw_data/ILSVRC2012_img_train.tar
check_status "Training data extraction"

# Extract individual class archives with progress
total_archives=$(ls *.tar | wc -l)
current=0
for f in *.tar; do
    ((current++))
    echo -ne "Extracting class archives: $current/$total_archives\r"
    d=$(basename $f .tar)
    mkdir -p $d
    cd $d
    tar -xf ../$f
    cd ..
    rm $f
done
echo -e "\nClass extraction completed!"

echo "Processing validation data..."
cd ../val
tar -xf ../../../../../raw_data/ILSVRC2012_img_val.tar
check_status "Validation data extraction"

# Download and run validation organization script
wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
check_status "Validation data organization"

# Verify data structure
echo "Verifying data structure..."
train_classes=$(ls ../train | wc -l)
val_classes=$(ls . | wc -l)

if [ "$train_classes" -ne 1000 ] || [ "$val_classes" -ne 1000 ]; then
    echo "Error: Expected 1000 classes, but found $train_classes in train and $val_classes in val"
    exit 1
fi

# Clean up
cd ../../../../../
rm -rf raw_data
check_status "Cleanup"

# Upload to S3 with progress
echo "Creating S3 bucket..."
aws s3 mb s3://${aws_bucket_name} || true

echo "Uploading to S3..."
aws s3 sync data/ILSVRC/Data/CLS-LOC/ s3://${aws_bucket_name}/ILSVRC/Data/CLS-LOC/ \
    --progress \
    --only-show-errors
check_status "S3 upload"

echo "Verifying S3 upload..."
s3_classes=$(aws s3 ls s3://${aws_bucket_name}/ILSVRC/Data/CLS-LOC/train/ | wc -l)
if [ "$s3_classes" -ne 1000 ]; then
    echo "Error: S3 upload verification failed. Expected 1000 classes, found $s3_classes"
    exit 1
fi

echo "ImageNet download and processing completed successfully!" 