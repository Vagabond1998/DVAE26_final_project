# DVAE26 Final Project
In this project we create an image classification model based on the MNIST dataset.

## Introduction and Problem Definition
Image recognition is a central problem in modern artificial intelligence and machine learning, with applications ranging from medical image analysis and autonomous driving to document processing and biometric systems. At its core, image recognition involves learning meaningful representations from raw pixel data in order to classify or interpret visual information. The complexity of this task arises from the high dimensionality of image data, variations in appearance, noise, and transformations such as rotation or scale changes.

Convolutional Neural Networks (CNNs) have become the dominant approach for image recognition tasks due to their ability to exploit spatial structure in images. By learning hierarchical feature representations through convolutional filters, CNNs can automatically extract relevant patterns such as edges, textures, and shapes, without the need for manually engineered features. As a result, CNN-based models have achieved state-of-the-art performance on a wide range of benchmark datasets and real-world applications.

In this project, the problem of image recognition is studied using the MNIST dataset, a well-known benchmark consisting of grayscale images of handwritten digits. Although MNIST is considered a relatively simple dataset by modern standards, it remains widely used for evaluating image classification pipelines due to its clean structure, standardized splits, and balanced class distribution. Importantly, MNIST allows a clear focus on the end-to-end machine learning workflow—including data handling, model design, training, evaluation, and deployment—without confounding factors introduced by large-scale or noisy datasets.

The objective of this project is to develop, train, and evaluate a convolutional neural network capable of accurately classifying handwritten digits from the MNIST dataset. Beyond achieving high classification accuracy, the project emphasizes software engineering best practices and reproducibility. This includes modular code design, explicit handling of data transformations, systematic evaluation using multiple metrics, and deployment of the trained model to a public platform.

From an engineering perspective, the project is structured as a complete machine learning pipeline. Raw image data is ingested and transformed into a suitable numerical representation, augmented to improve robustness, and then used to train a CNN model. Model performance is evaluated using standard classification metrics such as accuracy, precision, and recall. The final trained model is packaged and deployed to the Hugging Face platform, allowing external users to download and run inference using the trained network.

By using MNIST as a benchmark dataset, this project provides a controlled environment in which design decisions, training behavior, and evaluation results can be clearly analyzed and explained. The focus is therefore not only on model performance, but also on demonstrating a structured and reproducible approach to machine learning system development, in line with the objectives of the course.

## 2. Dataset Description and Data Quality Analysis
### 2.1 Dataset Overview

The dataset used in this project is the MNIST handwritten digit dataset, a widely adopted benchmark for image classification tasks. MNIST consists of grayscale images of handwritten digits ranging from 0 to 9. Each image has a fixed resolution of 28 × 28 pixels, resulting in a total of 784 pixel values per image. The dataset is divided into a training set containing 60,000 images and a test set containing 10,000 images.

Each image is associated with a single class label corresponding to the digit it represents. The dataset is approximately class-balanced, meaning that each digit class is represented by a similar number of samples. This balanced distribution simplifies evaluation and allows the use of standard classification metrics without the need for class weighting or resampling strategies.

MNIST images are stored as grayscale intensity values, where each pixel represents the brightness at a particular spatial location. This structure makes MNIST particularly suitable for convolutional neural networks, which are designed to exploit local spatial correlations in image data.

### 2.2 Data Quality Characteristics

MNIST is a curated and well-maintained benchmark dataset, and as such, it exhibits high data quality compared to many real-world datasets. No missing values, corrupted images, or inconsistent labels were observed during dataset inspection. All images share a consistent resolution and format, and labels are provided for every sample.

Unlike tabular datasets, image datasets do not typically require operations such as imputation or removal of invalid rows. Instead, data quality considerations for image data focus on factors such as visual clarity, label correctness, class balance, and robustness to natural variations in appearance. In the case of MNIST, the dataset’s controlled acquisition process and extensive prior use in the literature provide confidence in its label accuracy and overall integrity.

To verify dataset integrity within the project pipeline, the dataset was loaded programmatically and basic sanity checks were performed. These checks included verifying the total number of images in the training and test sets, confirming the expected image dimensions, and ensuring that label values fell within the valid range of 0 to 9.

### 2.3 Class Distribution and Balance

An important aspect of data quality for classification tasks is class distribution. Severe class imbalance can bias model training and distort evaluation metrics. The MNIST dataset exhibits an approximately uniform distribution across all ten digit classes in both the training and test sets. This balance reduces the risk of biased learning and enables the use of accuracy, precision, and recall as reliable evaluation metrics.

Because the dataset is balanced, no additional techniques such as oversampling, undersampling, or class-weighted loss functions were required. This allows model performance to be interpreted directly without adjusting for skewed class frequencies.

### 2.4 Normalization and Numerical Stability

Although MNIST does not require traditional data cleaning, numerical preprocessing is still necessary to ensure stable and efficient model training. Raw pixel values in MNIST are represented as integer intensities, which are not ideal for gradient-based optimization. To address this, images were normalized using the mean and standard deviation computed from the training set.

Normalization ensures that input values are centered and scaled consistently, which improves numerical stability during training and accelerates convergence. Importantly, normalization parameters were computed exclusively on the training data and then applied to both training and test sets, preventing information leakage from the test data into the training process.

### 2.5 Summary of Data Quality Considerations

In summary, the MNIST dataset provides a high-quality and well-structured foundation for image recognition experiments. While extensive data cleaning procedures were not required, careful attention was paid to dataset integrity, class balance, and numerical preprocessing. These steps ensure that observed model performance reflects the learning capability of the model rather than artifacts introduced by data quality issues.

The controlled nature of MNIST allows the project to focus on the design, training, and evaluation of the convolutional neural network, while still demonstrating appropriate data quality analysis practices relevant to image-based machine learning systems.

## Project Overview

## Dataset

## Data Quality Analysis

## Data Augmentation

## Model Architecture

## Training Procedure

## Evaluation

## Software Engineering Practices

## Deployment

## How to run the project locally

## Repository Structure

## References
