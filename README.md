# DVAE26 Final Project
In this project we create an image classification model based on the MNIST dataset.

## Introduction and Problem Definition
Image recognition is a central problem in modern artificial intelligence and machine learning, with applications ranging from medical image analysis and autonomous driving to document processing and biometric systems. At its core, image recognition involves learning meaningful representations from raw pixel data in order to classify or interpret visual information. The complexity of this task arises from the high dimensionality of image data, variations in appearance, noise, and transformations such as rotation or scale changes.

Convolutional Neural Networks (CNNs) have become the dominant approach for image recognition tasks due to their ability to exploit spatial structure in images. By learning hierarchical feature representations through convolutional filters, CNNs can automatically extract relevant patterns such as edges, textures, and shapes, without the need for manually engineered features. As a result, CNN-based models have achieved state-of-the-art performance on a wide range of benchmark datasets and real-world applications.

In this project, the problem of image recognition is studied using the MNIST dataset, a well-known benchmark consisting of grayscale images of handwritten digits. Although MNIST is considered a relatively simple dataset by modern standards, it remains widely used for evaluating image classification pipelines due to its clean structure, standardized splits, and balanced class distribution. Importantly, MNIST allows a clear focus on the end-to-end machine learning workflow—including data handling, model design, training, evaluation, and deployment—without confounding factors introduced by large-scale or noisy datasets.

The objective of this project is to develop, train, and evaluate a convolutional neural network capable of accurately classifying handwritten digits from the MNIST dataset. Beyond achieving high classification accuracy, the project emphasizes software engineering best practices and reproducibility. This includes modular code design, explicit handling of data transformations, systematic evaluation using multiple metrics, and deployment of the trained model to a public platform.

From an engineering perspective, the project is structured as a complete machine learning pipeline. Raw image data is ingested and transformed into a suitable numerical representation, augmented to improve robustness, and then used to train a CNN model. Model performance is evaluated using standard classification metrics such as accuracy, precision, and recall. The final trained model is packaged and deployed to the Hugging Face platform, allowing external users to download and run inference using the trained network.

By using MNIST as a benchmark dataset, this project provides a controlled environment in which design decisions, training behavior, and evaluation results can be clearly analyzed and explained. The focus is therefore not only on model performance, but also on demonstrating a structured and reproducible approach to machine learning system development, in line with the objectives of the course.

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
