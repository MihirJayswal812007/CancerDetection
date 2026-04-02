# References

## Datasets

* Histopathologic Cancer Detection Dataset (Kaggle)
* Breast Cancer Wisconsin Dataset (UCI ML Repository)

## Libraries

* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Streamlit

## Concepts

* Convolutional Neural Networks (CNN)
* Transfer Learning (MobileNetV2, ResNet50)
* Gradient Descent Optimization
* Binary Cross Entropy Loss
* Grad-CAM (Gradient-weighted Class Activation Mapping)

## Research Areas

* Medical Image Classification
* AI in Healthcare Diagnostics
* Explainable AI (XAI)

## Mathematical Concepts

* Partial Derivatives
* Gradient (∇L)
* Optimization using Gradient Descent
* Loss Minimization

## Tools

* Google Colab (GPU training)
* Kaggle (datasets)

## Google Colab Execution Guide

### Setup

1. Open Google Colab.
2. Enable GPU:
   Runtime -> Change runtime type -> GPU

### Install Dependencies

```python
!pip install tensorflow
```

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Project Structure in Drive

Place the project inside:

```text
MyDrive/cancer-detection/
```

### Run Training

```python
%cd /content/drive/MyDrive/cancer-detection
!python model/train.py
```

### Outputs

Saved in:

```text
outputs/
```

* model.h5
* loss.png
* accuracy.png

### Notes

* Do not train locally without GPU.
* Use Colab for all heavy computation.
