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

# Function to count files
count_files() {
    find "$1" -type f | wc -l
}

# Create target directories
echo "Creating target directories..."
mkdir -p data/ILSVRC/Data/CLS-LOC/{train,val}
check_status "Directory creation"

# Check if raw data exists
if [ ! -d "raw_data" ]; then
    echo "Error: raw_data directory not found"
    exit 1
fi

# First, let's check the structure
echo "Analyzing data structure..."
find raw_data -type d -ls

echo "Please confirm the correct paths for train and val directories:"
read -p "Train directory path (relative to raw_data/): " TRAIN_PATH
read -p "Val directory path (relative to raw_data/): " VAL_PATH

# Verify paths exist
if [ ! -d "raw_data/$TRAIN_PATH" ]; then
    echo "Error: Train directory not found at raw_data/$TRAIN_PATH"
    exit 1
fi

if [ ! -d "raw_data/$VAL_PATH" ]; then
    echo "Error: Validation directory not found at raw_data/$VAL_PATH"
    exit 1
fi

# Count initial files
echo "Counting initial files..."
INITIAL_TRAIN_COUNT=$(count_files "raw_data/$TRAIN_PATH")
INITIAL_VAL_COUNT=$(count_files "raw_data/$VAL_PATH")
echo "Found $INITIAL_TRAIN_COUNT training files and $INITIAL_VAL_COUNT validation files"

# Organize data using find with progress
echo "Organizing training data..."
cd "raw_data/$TRAIN_PATH"
total_classes=$(ls -1d */ 2>/dev/null | wc -l)
current=0
for class_dir in */; do
    if [ -d "$class_dir" ]; then
        ((current++))
        echo "Moving class: $class_dir ($current/$total_classes)"
        mkdir -p "../../../data/ILSVRC/Data/CLS-LOC/train/$class_dir"
        find "$class_dir" -type f -exec mv {} "../../../data/ILSVRC/Data/CLS-LOC/train/$class_dir" \;
    fi
done
cd ../../../

echo "Organizing validation data..."
cd "raw_data/$VAL_PATH"
total_classes=$(ls -1d */ 2>/dev/null | wc -l)
current=0
for class_dir in */; do
    if [ -d "$class_dir" ]; then
        ((current++))
        echo "Moving class: $class_dir ($current/$total_classes)"
        mkdir -p "../../../data/ILSVRC/Data/CLS-LOC/val/$class_dir"
        find "$class_dir" -type f -exec mv {} "../../../data/ILSVRC/Data/CLS-LOC/val/$class_dir" \;
    fi
done
cd ../../../

# Verify data organization
echo "Verifying data organization..."
FINAL_TRAIN_COUNT=$(count_files "data/ILSVRC/Data/CLS-LOC/train")
FINAL_VAL_COUNT=$(count_files "data/ILSVRC/Data/CLS-LOC/val")

echo "Initial counts: $INITIAL_TRAIN_COUNT training, $INITIAL_VAL_COUNT validation"
echo "Final counts: $FINAL_TRAIN_COUNT training, $FINAL_VAL_COUNT validation"

if [ "$INITIAL_TRAIN_COUNT" -ne "$FINAL_TRAIN_COUNT" ] || [ "$INITIAL_VAL_COUNT" -ne "$FINAL_VAL_COUNT" ]; then
    echo "Warning: File count mismatch. Please verify the data manually."
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

echo "Data organization and upload completed successfully!" 