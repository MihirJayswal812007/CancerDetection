# Project Context

## Project Title

AI-Based Cancer Detection with Gradient Optimization Visualization

## Objective

Build a machine learning system that detects cancer from medical images using deep learning and deploy it as a web application.

## Scope

* Binary image classification (Cancer vs Non-Cancer)
* Use transfer learning (MobileNetV2)
* Visualize model decisions using Grad-CAM
* Demonstrate vector calculus concepts through gradient descent and loss optimization

## Key Components

1. Data preprocessing (image resizing, normalization)
2. Model training using pretrained CNN
3. Evaluation using accuracy, precision, recall
4. Visualization of loss curves (gradient descent behavior)
5. Grad-CAM heatmap generation
6. Streamlit-based web application

## Constraints

* Must be completed within limited time
* Use GPU (Google Colab)
* Avoid training from scratch

## Expected Output

* Trained model
* Loss and accuracy graphs
* Grad-CAM visualizations
* Working Streamlit app

## Model Training Context

- Base model: MobileNetV2 pretrained on ImageNet
- Dataset: Kaggle Breast Histopathology Images (IDC)
- Image size: 96x96 pixels, RGB
- Task: Binary classification (IDC positive/negative)
- Current accuracy: 73.70%, Recall: 75.20%, Precision: 73.01%, F1: 74.09%
- Current threshold: hardcoded 0.65
- Training: Phase 1 (frozen base) + Phase 2 (fine-tune last 20 layers)
- Loss: binary_crossentropy, Optimizer: Adam
- Known issue: train/val accuracy gap suggests mild overfitting

## Upgrade Goals
- Target accuracy: 80%+
- Priority metric: Recall (minimize false negatives for cancer)
- Threshold should be computed from validation set using Youden's J
- Save threshold to model_threshold.json alongside model weights
- Training should run in Google Colab (no local GPU)