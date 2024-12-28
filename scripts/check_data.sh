#!/bin/bash

# Function to count files and show directory structure
check_directory() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo "Directory: $dir"
        echo "Number of files: $(find "$dir" -type f | wc -l)"
        echo "Number of directories: $(find "$dir" -type d | wc -l)"
        echo "First few subdirectories:"
        ls -l "$dir" | head -n 5
        echo "----------------------------------------"
    else
        echo "Directory $dir does not exist"
        echo "----------------------------------------"
    fi
}

echo "Checking all possible data locations..."

# Check raw_data locations
echo "1. Checking raw_data structure:"
check_directory "raw_data/ILSVRC/Data/CLS-LOC/train"
check_directory "raw_data/ILSVRC/Data/CLS-LOC/val"
check_directory "raw_data/data/ILSVRC/Data/CLS-LOC/train"
check_directory "raw_data/data/ILSVRC/Data/CLS-LOC/val"

# Check data locations
echo "2. Checking data structure:"
check_directory "data/ILSVRC/Data/CLS-LOC/train"
check_directory "data/ILSVRC/Data/CLS-LOC/val"

# Show total disk usage
echo "3. Disk usage summary:"
du -sh data/ILSVRC/Data/CLS-LOC/train 2>/dev/null || echo "No train directory in data/"
du -sh data/ILSVRC/Data/CLS-LOC/val 2>/dev/null || echo "No val directory in data/"
du -sh raw_data/ILSVRC/Data/CLS-LOC/train 2>/dev/null || echo "No train directory in raw_data/"
du -sh raw_data/ILSVRC/Data/CLS-LOC/val 2>/dev/null || echo "No val directory in raw_data/" 