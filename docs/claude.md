# Master Prompt for AI Code Generation

You are an expert machine learning engineer.

Build a complete cancer detection system using deep learning with the following requirements:

## Model

* Use MobileNetV2 pretrained on ImageNet
* Freeze base layers
* Add custom classification head:

  * GlobalAveragePooling
  * Dense (128, relu)
  * Dropout (0.5)
  * Dense (1, sigmoid)

## Training

* Binary classification (cancer vs non-cancer)
* Loss: binary crossentropy
* Optimizer: Adam
* Metrics: accuracy, precision, recall
* Epochs: 5–10
* Batch size: 32
* Use EarlyStopping and ModelCheckpoint

## Data

* Input images resized to 224x224
* Normalize pixel values
* Apply data augmentation (flip, rotation, zoom)

## Outputs

* Save trained model
* Plot loss and accuracy graphs

## Grad-CAM

* Implement Grad-CAM
* Generate heatmap highlighting important regions

## Streamlit App

* Upload image
* Show prediction
* Display confidence score
* Show Grad-CAM heatmap

## Code Quality

* Modular code
* Clear functions
* Comments where necessary

Generate complete working code for:

* training script
* grad-cam module
* streamlit app

## Training Code Rules
- Always use callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Always monitor val_recall as primary metric (medical AI priority)
- Never change model architecture unless explicitly asked
- Never modify Streamlit app.py during training tasks
- Always print class weights before training starts
- Save model as best_model.h5, not model.h5