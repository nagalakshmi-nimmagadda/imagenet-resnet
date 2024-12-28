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

# Function to show help
show_help() {
    echo "ImageNet Data Management Script"
    echo "Usage: ./scripts/manage_data.sh [command]"
    echo ""
    echo "Commands:"
    echo "  check        - Check current data structure and file counts"
    echo "  download     - Download ImageNet data from Kaggle (if not exists)"
    echo "  organize     - Organize existing data into proper structure"
    echo "  organize-val - Only organize validation data"
    echo "  upload       - Upload data to S3"
    echo "  clean        - Clean up temporary files"
    echo "  help        - Show this help message"
    echo ""
    echo "Example workflow:"
    echo "1. ./scripts/manage_data.sh check     (Check current data status)"
    echo "2. ./scripts/manage_data.sh organize  (Organize if needed)"
    echo "3. ./scripts/manage_data.sh upload    (Upload to S3)"
}

# Function to check data structure
check_data() {
    echo "Checking data structure..."
    
    # Check training data
    echo "Training data:"
    if [ -d "data/ILSVRC/Data/CLS-LOC/train" ]; then
        train_files=$(find data/ILSVRC/Data/CLS-LOC/train -type f | wc -l)
        train_classes=$(ls data/ILSVRC/Data/CLS-LOC/train | wc -l)
        echo "- Location: data/ILSVRC/Data/CLS-LOC/train"
        echo "- Files: $train_files"
        echo "- Classes: $train_classes"
        du -sh data/ILSVRC/Data/CLS-LOC/train
    else
        echo "- Not found in data directory"
    fi

    # Check validation data
    echo -e "\nValidation data:"
    if [ -d "raw_data/ILSVRC/Data/CLS-LOC/val" ]; then
        val_files=$(find raw_data/ILSVRC/Data/CLS-LOC/val -type f | wc -l)
        echo "- Location: raw_data/ILSVRC/Data/CLS-LOC/val"
        echo "- Files: $val_files"
        echo "- Status: Needs organization"
        du -sh raw_data/ILSVRC/Data/CLS-LOC/val
    elif [ -d "data/ILSVRC/Data/CLS-LOC/val" ]; then
        val_files=$(find data/ILSVRC/Data/CLS-LOC/val -type f | wc -l)
        val_classes=$(ls data/ILSVRC/Data/CLS-LOC/val | wc -l)
        echo "- Location: data/ILSVRC/Data/CLS-LOC/val"
        echo "- Files: $val_files"
        echo "- Classes: $val_classes"
        echo "- Status: Organized"
        du -sh data/ILSVRC/Data/CLS-LOC/val
    else
        echo "- Not found"
    fi

    # Provide recommendations
    echo -e "\nRecommendations:"
    if [ ! -d "data/ILSVRC/Data/CLS-LOC/train" ]; then
        echo "- Training data missing. Run 'download' if needed."
    fi
    if [ -d "raw_data/ILSVRC/Data/CLS-LOC/val" ] && [ ! -d "data/ILSVRC/Data/CLS-LOC/val" ]; then
        echo "- Validation data needs organization. Run 'organize-val'."
    fi
    if [ ! -d "raw_data/ILSVRC/Data/CLS-LOC/val" ] && [ ! -d "data/ILSVRC/Data/CLS-LOC/val" ]; then
        echo "- Validation data missing. Run 'download' if needed."
    fi
}

# Function to download data
download_data() {
    # Check if data already exists
    if [ -d "data/ILSVRC/Data/CLS-LOC/train" ] || [ -d "raw_data/ILSVRC/Data/CLS-LOC/train" ]; then
        echo "Training data already exists. Run 'check' to see current status."
        read -p "Do you want to proceed with download anyway? (y/N) " confirm
        if [[ $confirm != [yY] ]]; then
            echo "Download cancelled."
            return
        fi
    fi
    
    echo "Downloading ImageNet data..."
    
    # Install dependencies
    sudo apt-get update
    sudo apt-get install -y python3-pip unzip
    pip install --upgrade kaggle
    
    # Setup Kaggle credentials
    mkdir -p ~/.kaggle
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo "Please enter your Kaggle credentials:"
        read -p "Username: " username
        read -p "Key: " key
        echo "{\"username\":\"$username\",\"key\":\"$key\"}" > ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json
    fi
    
    # Download data
    mkdir -p raw_data
    kaggle competitions download -c imagenet-object-localization-challenge -p raw_data
    unzip raw_data/imagenet-object-localization-challenge.zip -d raw_data/
}

# Function to organize validation data
organize_val() {
    echo "Organizing validation data..."
    
    # Verify validation files exist
    val_dir="data/ILSVRC/Data/CLS-LOC/val"
    if [ ! -d "$val_dir" ]; then
        echo "Error: Validation directory not found at $val_dir"
        exit 1
    fi
    
    total_files=$(find "$val_dir" -maxdepth 1 -name "*.JPEG" | wc -l)
    echo "Found $total_files validation files to organize"
    
    if [ "$total_files" -eq 0 ]; then
        echo "Error: No validation files found in $val_dir"
        exit 1
    fi
    
    # Download validation preparation script
    if [ ! -f "valprep.sh" ]; then
        echo "Downloading validation preparation script..."
        wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh -O valprep.sh
        chmod +x valprep.sh
        
        # Verify download
        if [ ! -f "valprep.sh" ]; then
            echo "Error: Failed to download valprep.sh"
            exit 1
        fi
    fi
    
    # Run validation preparation script
    echo "Organizing validation files into classes..."
    cd "$val_dir"
    ../../../../../valprep.sh
    cd ../../../../../
    
    # Verify organization
    val_classes=$(ls "$val_dir" | wc -l)
    val_files=$(find "$val_dir" -type f | wc -l)
    echo "Found $val_classes classes with $val_files files"
    
    if [ "$val_classes" -lt 900 ]; then
        echo "Warning: Expected 1000 classes, found $val_classes"
        return 1
    fi
    
    echo "Validation data organization completed successfully!"
}

# Function to organize training data
organize_train() {
    echo "Checking training data organization..."
    
    # Check if raw training data exists
    if [ ! -d "raw_data/ILSVRC/Data/CLS-LOC/train" ]; then
        echo "No raw training data found to organize"
        return 0
    fi
    
    # Create target directory
    mkdir -p data/ILSVRC/Data/CLS-LOC/train
    
    # Move class directories
    echo "Moving training class directories..."
    total_classes=$(ls raw_data/ILSVRC/Data/CLS-LOC/train | wc -l)
    current=0
    
    for class_dir in raw_data/ILSVRC/Data/CLS-LOC/train/*/; do
        ((current++))
        class_name=$(basename "$class_dir")
        echo -ne "Moving class $current/$total_classes: $class_name\r"
        
        # Move entire class directory
        mv "$class_dir" "data/ILSVRC/Data/CLS-LOC/train/"
    done
    echo -e "\nFinished moving $current classes"
    
    # Verify organization
    train_classes=$(ls data/ILSVRC/Data/CLS-LOC/train | wc -l)
    train_files=$(find data/ILSVRC/Data/CLS-LOC/train -type f | wc -l)
    echo "Found $train_classes classes with $train_files total files"
    
    if [ "$train_classes" -ne 1000 ]; then
        echo "Warning: Expected 1000 classes, found $train_classes"
        return 1
    fi
    
    echo "Training data organization completed successfully!"
}

# Function to upload to S3
upload_to_s3() {
    # Check AWS credentials
    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        echo "Error: AWS credentials not found"
        echo "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in utils/config.sh"
        exit 1
    fi

    # Check if data exists before upload
    if [ ! -d "data/ILSVRC/Data/CLS-LOC/train" ] || [ ! -d "data/ILSVRC/Data/CLS-LOC/val" ]; then
        echo "Error: Data not found in expected location. Run 'check' to verify data status."
        exit 1
    fi

    echo "Preparing to upload to S3..."
    echo "This will upload data to: s3://${aws_bucket_name}/ILSVRC/Data/CLS-LOC/"
    read -p "Do you want to proceed? (y/N) " confirm
    if [[ $confirm != [yY] ]]; then
        echo "Upload cancelled."
        return
    fi

    # Create bucket if it doesn't exist
    if ! aws s3 ls "s3://${aws_bucket_name}" 2>&1 > /dev/null; then
        echo "Creating bucket ${aws_bucket_name}..."
        aws s3 mb "s3://${aws_bucket_name}"
    fi

    # Upload data
    echo "Uploading data..."
    aws s3 sync data/ILSVRC/Data/CLS-LOC/ "s3://${aws_bucket_name}/ILSVRC/Data/CLS-LOC/" \
        --storage-class STANDARD \
        --only-show-errors
}

# Main command processing
case "$1" in
    "check")
        check_data
        ;;
    "download")
        download_data
        ;;
    "organize")
        organize_train
        organize_val
        ;;
    "organize-train")
        organize_train
        ;;
    "organize-val")
        organize_val
        ;;
    "upload")
        upload_to_s3
        ;;
    "clean")
        read -p "This will remove raw_data directory. Are you sure? (y/N) " confirm
        if [[ $confirm == [yY] ]]; then
            rm -rf raw_data
            echo "Cleanup completed."
        else
            echo "Cleanup cancelled."
        fi
        ;;
    "help"|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 