# ImageNet Training and Deployment Pipeline 🚀

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Spaces-Deployment-yellow.svg)](https://huggingface.co/spaces)
[![AWS](https://img.shields.io/badge/AWS-EC2%20Optimized-orange.svg)](https://aws.amazon.com/ec2/)

A production-ready pipeline for training ResNet50 on ImageNet and deploying to HuggingFace Spaces, optimized for AWS EC2 GPU instances.

## 📊 Performance

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 79.26% |
| Top-5 Accuracy | 94.51% |
| Training Time | ~24 hours (g6.12xlarge) |
| Model Size | 98 MB |

## 🌟 Features

- **Optimized Training Pipeline**
  - Mixed precision training
  - Gradient accumulation
  - Multi-GPU support
  - Automatic checkpointing

- **Cost-Efficient Infrastructure**
  - AWS Spot Instance support
  - Automatic spot termination handling
  - S3 checkpoint synchronization
  - Resource monitoring

- **Production Deployment**
  - HuggingFace Spaces integration
  - Gradio web interface
  - Automated deployment pipeline
  - Version control with git-lfs

## 🏗️ Project Structure


Imagenet-resnet/
├── checkpoints/ # Model checkpoints
│ └── last.ckpt # Latest checkpoint
├── configs/ # Configuration files
│ ├── config.yaml # Training configuration
│ └── aws_config.yaml # AWS configuration
├── data/ # Dataset management
│ ├── init.py
│ ├── dataset.py # ImageNet dataset
│ └── transforms.py # Data augmentation
├── deployment/ # HuggingFace deployment
│ ├── app.py # Gradio interface
│ ├── requirements.txt # Dependencies
│ └── scripts/ # Deployment scripts
├── logs/ # Training logs
├── scripts/ # Utility scripts
│ ├── setup_instance.sh
│ ├── train_spot.sh
│ └── monitor.sh
└── requirements.txt # Project dependencies


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
