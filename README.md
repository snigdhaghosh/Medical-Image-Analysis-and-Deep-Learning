# Medical Image Diagnostics using Machine Learning & Deep Learning
Can sophisticated computer vision models provide robust, reliable, and scalable support for medical diagnostics, enhancing the accuracy, efficiency, and accessibility of healthcare delivery?

This is a group project developed as part of the final project.

## Overview
Medical imaging has revolutionized healthcare by providing crucial insights from disease detection to treatment planning. However, the rapid growth in the use of medical imaging technologies presents several substantial challenges that need to be addressed to fully realize their potential.

This project explores the application of computer vision techniques to improve medical image diagnostics. We implemented and compared multiple machine learning and deep learning approaches, ranging from traditional feature extraction methods to advanced deep learning architectures. Our goal was to enhance the accuracy, efficiency, and scalability of automated medical diagnostics using datasets like Medical MNIST and COVID-19 Radiography.

## Project Structure
1️⃣ Traditional Machine Learning Approaches
Feature extraction using Histogram of Oriented Gradients (HoG) and Scale-Invariant Feature Transform (SIFT).
Classification using Support Vector Machines (SVMs) and Decision Trees with Bagging & Boosting Ensembles.
Bag of Words (BoW) model for feature representation.

2️⃣ Deep Learning Approaches (CNNs)
Designed two CNN architectures:
- Simple (Base) Model – Lightweight and efficient, trained with Adam & Nadam optimizers.
- Bigger (Advanced) Model – A deeper architecture with L2 regularization and dropout layers for improved generalization.

3️⃣ Transfer Learning with ResNet-50
Fine-tuned ResNet-50, a 50-layer residual network, pre-trained on ImageNet.
Unfrozen the last 40 layers for feature adaptation, optimized with Categorical Cross-Entropy and ReduceLROnPlateau.

## Datasets
- Medical MNIST – 60,000 grayscale images across six medical imaging classes.
- COVID-19 Radiography Dataset – 40,000 X-ray images across four classes: Normal, COVID, Viral Pneumonia, and Lung Opacity.

## Key Results & Findings

| Model |  Medical MNIST Accuracy   | COVID-19 Radiography Accuracy  |
| :-----: | :---: | :---: |
| SVM + HoG | 99.5%   | 88.25%   |
| Simple CNN | 99.86%   | 92.76%   |
| Bigger CNN | 98.29%   | 57.40%   |
|  ResNet-50 (Transfer Learning) | 99.96%   | 97.42%   |

- Traditional methods performed well on simpler datasets but struggled with complex medical images.
- CNNs provided significant improvements, but deeper architectures required more tuning.
- ResNet-50 with fine-tuning achieved the highest accuracy, demonstrating the power of transfer learning in medical diagnostics.




## Challenges & Future Work
🚀 Challenges
Class imbalance – Mitigated using weighted loss functions and data augmentation.
Overfitting – Addressed with L2 regularization, dropout layers, and early stopping.
Computational cost – Optimized using pre-trained models and transfer learning.
🔬 Future Work
Extend to other medical imaging modalities (MRI, CT scans).
Implement Vision Transformers (ViTs) for improved feature extraction.
Develop explainable AI (XAI) methods to enhance interpretability for clinicians.


## References
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. arXiv:1512.03385.
- COVID-19 Radiography Dataset – IEEE Access, 2020.
- Medical MNIST Dataset – Kaggle repository.

  
