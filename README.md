# Medical Image Diagnostics using Machine Learning & Deep Learning
Can sophisticated computer vision models provide robust, reliable, and scalable support for medical diagnostics, enhancing the accuracy, efficiency, and accessibility of healthcare delivery?

## Overview
This project explores the application of computer vision techniques to improve medical image diagnostics. We implemented and compared multiple machine learning and deep learning approaches, ranging from traditional feature extraction methods to advanced deep learning architectures. Our goal was to enhance the accuracy, efficiency, and scalability of automated medical diagnostics using datasets like Medical MNIST and COVID-19 Radiography.

This is a group project developed as part of the final project.

## Project Structure
1️⃣ Traditional Machine Learning Approaches
Feature extraction using Histogram of Oriented Gradients (HoG) and Scale-Invariant Feature Transform (SIFT).
Classification using Support Vector Machines (SVMs) and Decision Trees with Bagging & Boosting Ensembles.
Bag of Words (BoW) model for feature representation.
2️⃣ Deep Learning Approaches (CNNs)
Designed two CNN architectures:
Simple (Base) Model – Lightweight and efficient, trained with Adam & Nadam optimizers.
Bigger (Advanced) Model – A deeper architecture with L2 regularization and dropout layers for improved generalization.
3️⃣ Transfer Learning with ResNet-50
Fine-tuned ResNet-50, a 50-layer residual network, pre-trained on ImageNet.
Unfrozen the last 40 layers for feature adaptation, optimized with Categorical Cross-Entropy and ReduceLROnPlateau.




