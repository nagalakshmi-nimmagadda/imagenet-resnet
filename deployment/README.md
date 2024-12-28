---
title: ImageNet Classifier
emoji: 🖼️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
---

# ImageNet Classifier

This is a ResNet50 model trained on ImageNet, fine-tuned for better performance. The model achieves:
- Top-1 Accuracy: 79.26%
- Top-5 Accuracy: 94.51%

## Usage
1. Upload an image
2. Get predictions for the top 5 classes

## Model Details
- Architecture: ResNet50
- Training Dataset: ImageNet
- Input Size: 224x224
- Number of Classes: 1000 