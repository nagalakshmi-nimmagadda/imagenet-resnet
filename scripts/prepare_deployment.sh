#!/bin/bash
set -e

# Create deployment structure
echo "Creating deployment structure..."
mkdir -p deployment/examples

# Copy latest checkpoint
echo "Copying best model checkpoint..."
cp checkpoints/last.ckpt deployment/model.ckpt

# Convert model if needed
echo "Converting model..."
python deployment/convert_model.py

# Copy example images
echo "Setting up examples..."
if [ ! -d "deployment/examples" ]; then
    mkdir -p deployment/examples
    wget -P deployment/examples https://raw.githubusercontent.com/pytorch/examples/main/imagenet/cat.jpg
    wget -P deployment/examples https://raw.githubusercontent.com/pytorch/examples/main/imagenet/dog.jpg
    wget -P deployment/examples https://raw.githubusercontent.com/pytorch/examples/main/imagenet/bird.jpg
fi

# Download ImageNet classes
echo "Downloading ImageNet classes..."
python scripts/get_imagenet_classes.py

# Create HuggingFace Space
echo "Setting up HuggingFace Space..."
huggingface-cli login
huggingface-cli repo create imagenet-classifier --type space

# Clone and setup Space
git clone https://huggingface.co/spaces/$USER/imagenet-classifier
cp -r deployment/* imagenet-classifier/
cd imagenet-classifier

# Push to HuggingFace
git add .
git commit -m "Initial deployment"
git push 