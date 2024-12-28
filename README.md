# ImageNet Training and Deployment Guide

This repository contains code for training ResNet50 on ImageNet using AWS EC2 instances and deploying to HuggingFace Spaces.

## Current Results
- Top-1 Accuracy: 79.26%
- Top-5 Accuracy: 94.51%

## Prerequisites
- AWS Account with EC2 access
- NVIDIA GPU instance (g6.12xlarge recommended)
- HuggingFace account
- ImageNet dataset

## Directory Structure 


├── checkpoints/ # Model checkpoints
├── data/ # Dataset and data loading
├── deployment/ # HuggingFace deployment files
├── logs/ # Training logs
├── scripts/ # Utility scripts
└── configs/ # Configuration files


## Step-by-Step Guide

### 1. Initial Setup

Clone the repository
git clone <repository-url>
cd imagenet-resnet-1
Create virtual environment
python -m venv env
source env/bin/activate
Install requirements
pip install -r requirements.txt



### 2. Instance Setup

Make setup script executable
chmod +x scripts/setup_instance.sh

Run setup script
sudo ./scripts/setup_instance.sh

Key configurations:
- NVIDIA drivers
- CUDA toolkit
- System optimizations
- Dataset mounting

### 3. Dataset Preparation

1. Mount ImageNet dataset:
Create data directories
mkdir -p data/imagenet/{train,val}
Mount dataset
aws s3 sync s3://your-bucket/imagenet/train data/imagenet/train
aws s3 sync s3://your-bucket/imagenet/val data/imagenet/val
2. Verify dataset structure:

data/imagenet/
├── train/
│ ├── n01440764/
│ ├── n01443537/
│ └── ...
└── val/
├── n01440764/
├── n01443537/
└── ...


### 4. Training

1. Configure training:
configs/config.yaml
data:
train_dir: "data/imagenet/train"
val_dir: "data/imagenet/val"
batch_size: 128
num_workers: 8


2. Start training:

Make training script executable
chmod +x start_training.sh
Start training
./start_training.sh


3. Monitor training:
In a separate terminal
./scripts/monitor_training.sh

### 5. Spot Instance Management

For cost optimization, we use spot instances:

Make spot management script executable
chmod +x scripts/manage_spot.sh
Start spot instance monitoring
./scripts/manage_spot.sh


The script handles:
- Spot termination notices
- Checkpoint saving
- Training resumption

### 6. Deployment to HuggingFace

1. Prepare deployment:

Create deployment directory
mkdir -p deployment
Copy necessary files
cp -r app.py requirements.txt README.md deployment/

2. Setup HuggingFace:
Install HuggingFace CLI
pip install huggingface_hub
Login to HuggingFace
huggingface-cli login
Create space
huggingface-cli repo create imagenet-classifier --type space

3. Deploy:

Run deployment script
cd deployment
python scripts/prepare_deployment.py