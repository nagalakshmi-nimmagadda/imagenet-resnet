#!/bin/bash
set -e

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

echo "Organizing validation data..."

# Download validation labels if not present
if [ ! -f "val_labels.txt" ]; then
    echo "Downloading validation labels..."
    wget -q https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt -O val_labels.txt
    check_status "Downloading validation labels"
fi

# Create validation directories
cd data/ILSVRC/Data/CLS-LOC/val

# Create class directories based on training set structure
echo "Creating class directories..."
for class_dir in ../train/*/; do
    class_name=$(basename "$class_dir")
    mkdir -p "$class_name"
done

# Move validation files to appropriate directories
echo "Moving validation files..."
total_files=$(ls -1 ../../../raw_data/ILSVRC/Data/CLS-LOC/val/*.JPEG | wc -l)
current=0

while IFS= read -r line; do
    ((current++))
    file_num=$(printf "%08d" $current)
    file_name="ILSVRC2012_val_${file_num}.JPEG"
    class_name=$line
    
    echo -ne "Processing file $current/$total_files\r"
    
    if [ -f "../../../raw_data/ILSVRC/Data/CLS-LOC/val/$file_name" ]; then
        mv "../../../raw_data/ILSVRC/Data/CLS-LOC/val/$file_name" "$class_name/"
    fi
done < val_labels.txt

echo -e "\nValidation data organization completed!"

# Clean up
cd ../../../../../
rm -f val_labels.txt

# Verify counts
val_files=$(find data/ILSVRC/Data/CLS-LOC/val -type f | wc -l)
echo "Total validation files organized: $val_files"

if [ "$val_files" -ne 50000 ]; then
    echo "Warning: Expected 50000 validation files, found $val_files"
    exit 1
fi

echo "Validation data organization completed successfully!" 