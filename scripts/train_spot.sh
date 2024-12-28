#!/bin/bash
set -e

# Set up S3 bucket for checkpoints
BUCKET_NAME="your-bucket-name"
CHECKPOINT_DIR="checkpoints"

# Sync checkpoints from S3 if they exist
aws s3 sync s3://${BUCKET_NAME}/${CHECKPOINT_DIR} ${CHECKPOINT_DIR}

# Start spot termination checker in background
./scripts/manage_spot.sh check_termination &

# Start training
./start_training.sh

# Sync checkpoints to S3 after training
aws s3 sync ${CHECKPOINT_DIR} s3://${BUCKET_NAME}/${CHECKPOINT_DIR} 