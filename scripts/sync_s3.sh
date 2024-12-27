#!/bin/bash
set -e

# Source config
source utils/config.sh

# Function to show usage
usage() {
    echo "Usage: $0 {upload|download}"
    exit 1
}

# Check arguments
if [ "$#" -ne 1 ]; then
    usage
fi

ACTION=$1

case "$ACTION" in
    "upload")
        echo "Uploading data to S3 bucket: $aws_bucket_name..."
        aws s3 sync data/ILSVRC/Data/CLS-LOC/ s3://$aws_bucket_name/ILSVRC/Data/CLS-LOC/ --delete
        aws s3 sync checkpoints/ s3://$aws_bucket_name/checkpoints/ --delete
        echo "Upload completed!"
        ;;
    "download")
        echo "Downloading data from S3 bucket: $aws_bucket_name..."
        aws s3 sync s3://$aws_bucket_name/ILSVRC/Data/CLS-LOC/ data/ILSVRC/Data/CLS-LOC/ --delete
        aws s3 sync s3://$aws_bucket_name/checkpoints/ checkpoints/ --delete
        echo "Download completed!"
        ;;
    *)
        usage
        ;;
esac 