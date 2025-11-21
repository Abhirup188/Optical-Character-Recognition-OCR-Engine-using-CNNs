Optical Character Recognition (OCR) Engine using CNNs

Project Overview:

In industrial automation and digitization, systems need to automatically read numerical data (like serial numbers, meter readings, or handwritten logs) from images with high reliability. This project implements a robust Convolutional Neural Network (CNN) architecture to identify handwritten digits and images with human-level accuracy. This architecture serves as the foundation for reading serial codes on machinery in automated inspection lines.

The Engineering Problem:

Traditional computer vision methods often struggle with variations in handwriting, lighting, and orientation. A major challenge in automated data entry and inspection is building a system that can generalize from training data to unseen, real-world examples without extensive manual feature engineering.

The Solution:

Developed a custom CNN architecture capable of learning hierarchical feature representations from raw pixel data. By leveraging deep learning, the system automatically extracts spatial features (edges, textures, shapes) to classify images into distinct categories with high confidence.

Key Features:

Custom CNN Architecture: Implemented multiple convolution, pooling, and dropout layers to extract spatial features from images while preventing overfitting.

High Accuracy: Achieved >98% accuracy on the test dataset, demonstrating robust feature learning and generalization.

Scalability: The model architecture is modular and designed to be extensible to full alphanumeric character recognition for diverse industrial applications.

GPU Acceleration: Optimized for CUDA 12.1, enabling rapid training and inference on NVIDIA hardware.

Tech Stack:

Core Framework: PyTorch (with CUDA 12.1 support)

Vision Library: Torchvision

Data Processing: NumPy, Pillow (PIL)

Visualization: Matplotlib

Environment: Python 3.x, Jupyter Notebook

Model Architecture:

The solution utilizes a sequential CNN model with the following layers:

Convolutional Layers: To capture spatial features like edges and corners.

ReLU Activation: To introduce non-linearity for learning complex patterns.

Max Pooling: To reduce dimensionality and computational load while retaining key features.

Fully Connected Layers: To map extracted features to final class probabilities.

Getting Started:

Prerequisites

Python 3.8+

NVIDIA GPU with CUDA 12.1 (Recommended for training)

Installation

Clone the repository:

git clone [https://github.com/Abhirup188/Optical-Character-Recognition-OCR-Engine-using-CNNs.git](https://github.com/Abhirup188/Optical-Character-Recognition-OCR-Engine-using-CNNs.git)
cd Optical-Character-Recognition-OCR-Engine-using-CNNs


Install dependencies (optimized for CUDA 12.1):

pip install -r requirements.txt


Running the Model:

Open the Jupyter Notebook to view the training pipeline and inference results:

jupyter notebook image_classification_cnn.ipynb


Results:

Training Accuracy: >98%

Test Accuracy: Validated on unseen test data with consistent performance.

Visuals: See the notebook for confusion matrices and sample predictions.

Built by [Abhirup Chakraborty] - Mechanical Engineering & Robotics Specialist
