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

## 3. Exploratory Data Analysis (EDA)
### 3.1 Purpose of Exploratory Data Analysis for Image Data

Exploratory Data Analysis (EDA) is a critical step in any machine learning pipeline, as it provides insights into the structure, variability, and potential challenges of the dataset before model training. For image-based datasets, EDA differs from traditional tabular data analysis. Rather than focusing on statistical summaries of individual features, image EDA emphasizes visual inspection, dimensionality understanding, and assessment of intra-class and inter-class variability.

The primary objectives of EDA in this project are to verify that the dataset has been loaded correctly, to understand the visual characteristics of handwritten digits, and to identify sources of variation that the model must learn to handle.

### 3.2 Dataset Shape and Dimensionality

Each MNIST image consists of 28 × 28 grayscale pixels, resulting in a two-dimensional array per sample. When loaded into the training pipeline, images are converted into tensors with shape (1, 28, 28), where the leading dimension corresponds to the single grayscale channel.

The full training dataset therefore has the shape (60000, 1, 28, 28), while the test dataset has the shape (10000, 1, 28, 28). This confirms that the dataset has been loaded correctly and matches the expected structure defined by the MNIST specification.

Understanding the dimensionality of the input data is essential for designing the convolutional neural network architecture, as convolutional layers operate directly on the spatial structure of the input tensors.

### 3.3 Visual Inspection of Samples

A key component of EDA for image data is visual inspection of representative samples from each class. Random samples from the dataset reveal significant variation in handwriting style, stroke thickness, digit size, and orientation. For example, the digit “1” may appear as a simple vertical line in some samples, while in others it includes a serif or diagonal stroke. Similarly, digits such as “4” and “9” can exhibit structural similarities that make them more challenging to distinguish.

Visual inspection confirms that the dataset contains a wide range of realistic handwriting styles. This variability motivates the use of data augmentation techniques during training to improve the model’s robustness to small transformations and stylistic differences.

### 3.4 Intra-Class and Inter-Class Variability

EDA also highlights two important characteristics of the dataset:

- Intra-class variability: Samples belonging to the same digit class can differ substantially in appearance due to differences in handwriting style, stroke pressure, and orientation.

- Inter-class similarity: Certain digit pairs, such as “3” and “5” or “4” and “9”, may share visual similarities that increase the risk of misclassification.

These observations justify the use of convolutional neural networks, which are well-suited for learning hierarchical features that can capture both local patterns (such as edges and curves) and global digit structure.

### 3.5 Feature Engineering Considerations

Unlike classical machine learning approaches, convolutional neural networks do not require explicit manual feature engineering. Instead, the network learns relevant features directly from the pixel data during training. As a result, no handcrafted features such as edge detectors or shape descriptors were introduced.

However, preprocessing steps such as normalization and data augmentation can be viewed as forms of feature preparation that improve learning stability and generalization. These steps are discussed in detail in the following section.

### 3.6 Summary of EDA Findings

Exploratory Data Analysis confirmed that the MNIST dataset is correctly structured, visually diverse, and suitable for training a convolutional neural network. The observed variability in handwriting styles underscores the importance of robust model design and appropriate data augmentation. Additionally, the absence of severe class imbalance or corrupted samples allows the focus to remain on model architecture and training strategy rather than extensive data cleaning.

The insights gained from EDA directly inform the preprocessing and modeling decisions described in subsequent sections.

## 4. Data Transformation and Augmentation
### 4.1 Role of Data Transformation in Image Recognition

Before images can be used as input to a convolutional neural network, they must be transformed into a numerical format suitable for gradient-based optimization. In this project, data transformation serves two primary purposes: converting raw image data into tensors that can be processed by the model, and normalizing input values to ensure stable and efficient training.

Raw MNIST images are stored as grayscale intensity values. These images are first converted into tensors and reshaped to include an explicit channel dimension. This transformation results in tensors of shape (1, 28, 28), which is the expected input format for the convolutional layers used in the model architecture.

### 4.2 Normalization for Numerical Stability

Normalization is a crucial preprocessing step in neural network training. Without normalization, input values can vary across a wide numerical range, which may lead to unstable gradients and slow convergence during training. To address this, pixel values are normalized using the mean and standard deviation computed from the MNIST training set.

By centering and scaling the input distribution, normalization improves numerical stability and allows the optimizer to operate more effectively. Importantly, normalization parameters are computed using only the training data and then applied consistently to both training and test datasets. This prevents information leakage from the test set into the training process and ensures a fair evaluation of model performance.

### 4.3 Data Augmentation Strategy

Data augmentation is employed to enhance the diversity of the training dataset and improve the model’s ability to generalize to unseen samples. Although MNIST is a clean and curated dataset, handwritten digits exhibit natural variations in orientation and writing style. Without augmentation, a model may overfit to the specific orientations and patterns present in the training set.

In this project, data augmentation is implemented using random rotations applied to training images. Small rotations simulate realistic variations in handwriting, such as slightly tilted digits, without altering the semantic meaning of the image. This encourages the model to learn rotation-invariant features rather than memorizing specific pixel configurations.

### 4.4 Separation of Training and Test Transformations

A key design decision in the preprocessing pipeline is the strict separation of transformations applied to the training and test datasets. Data augmentation is applied exclusively to the training data, while the test data undergoes only deterministic transformations such as tensor conversion and normalization.

This separation ensures that evaluation metrics reflect the model’s performance on unmodified data and prevents artificial inflation of test accuracy. Applying augmentation to the test set would introduce randomness into the evaluation process and compromise the reliability of the reported results.

4.5 Implementation Considerations

The transformation and augmentation logic is implemented in a modular and reusable manner. Separate transformation pipelines are defined for training and testing, allowing preprocessing behavior to be adjusted independently without modifying the rest of the training code. This modular design improves code clarity, simplifies experimentation, and supports reproducibility.

By encapsulating transformations within dedicated functions, the preprocessing pipeline remains transparent and easy to audit, which is particularly important in an academic and engineering-focused context.

4.6 Summary of Data Preparation Approach

In summary, data transformation and augmentation play a central role in preparing the MNIST dataset for convolutional neural network training. Through tensor conversion, normalization, and carefully chosen augmentation techniques, the input data is made numerically stable and more representative of real-world handwriting variability. These preprocessing steps directly support the model’s ability to learn robust and generalizable representations from image data.

## 5. Model Architecture
### 5.1 Motivation for Using a Convolutional Neural Network

Convolutional Neural Networks (CNNs) are specifically designed to process grid-structured data such as images. Unlike fully connected networks, CNNs exploit the spatial locality and hierarchical structure of image data through the use of convolutional filters and shared weights. This makes them particularly effective for image recognition tasks, where local patterns such as edges, corners, and curves combine to form higher-level structures.

Given the two-dimensional nature of MNIST images and the need to capture spatial relationships between pixels, a CNN is a natural and well-established choice for this task.

### 5.2 Overall Network Structure

The model used in this project is a custom-designed CNN implemented in PyTorch. It follows a relatively simple architecture consisting of multiple convolutional layers followed by pooling operations and fully connected layers. This design reflects a balance between expressive power and computational efficiency, which is appropriate for the MNIST dataset.

At a high level, the network is composed of the following components:

1. Convolutional layers to extract local spatial features from input images.
2. Non-linear activation functions to introduce non-linearity and enable the learning of complex patterns.
3. Pooling layers to reduce spatial dimensionality and provide translation invariance.
4. Fully connected layers to map learned feature representations to class scores.

### 5.3 Convolutional Feature Extraction

The initial layers of the network consist of convolutional layers that operate directly on the input image tensors. These layers apply learnable filters across the spatial dimensions of the image to detect local patterns such as edges and simple shapes. As depth increases, subsequent convolutional layers learn progressively more abstract features, combining lower-level patterns into higher-level representations of digit structure.

Each convolutional layer is followed by a non-linear activation function, allowing the network to model complex, non-linear relationships in the data. This hierarchical feature extraction process is central to the effectiveness of CNNs in image recognition tasks.

### 5.4 Pooling and Dimensionality Reduction

Pooling layers are interleaved with convolutional layers to reduce the spatial resolution of feature maps. By summarizing local neighborhoods, pooling helps reduce computational complexity and provides a degree of invariance to small translations in the input image.

This is particularly useful for handwritten digit recognition, where the precise position of a stroke may vary slightly between samples without changing the digit’s identity. Pooling allows the network to focus on the presence of features rather than their exact location.

### 5.5 Fully Connected Classification Layers

After convolutional feature extraction, the resulting feature maps are flattened and passed to fully connected layers. These layers integrate information from all extracted features and produce a set of output scores, one for each digit class.

The final layer of the network outputs raw class scores, often referred to as logits. These logits are later converted into class probabilities during evaluation using a softmax function. The predicted digit corresponds to the class with the highest score.

### 5.6 Architectural Simplicity and Design Rationale

The chosen architecture is intentionally kept simple. MNIST is a relatively small and clean dataset, and more complex architectures would provide limited performance gains while increasing training time and implementation complexity. A compact CNN is sufficient to achieve high accuracy on this dataset while maintaining interpretability and ease of analysis.

This design choice aligns with the goals of the project, which emphasize clarity, reproducibility, and software engineering best practices over maximizing performance through overly complex models.

### 5.7 Summary of Model Architecture

In summary, the model architecture leverages the core strengths of convolutional neural networks to perform image classification effectively. Through convolutional feature extraction, pooling-based dimensionality reduction, and fully connected classification layers, the network learns hierarchical representations of handwritten digits. The simplicity of the architecture ensures efficient training and clear interpretability, making it well-suited for both educational purposes and systematic analysis.

## 6. Training Procedure and Hyperparameter Selection
### 6.1 Training Objective and Loss Function

The objective of the training process is to learn model parameters that minimize classification error on handwritten digit images. This is formulated as a multi-class classification problem with ten output classes corresponding to digits 0 through 9. The network is trained using the cross-entropy loss function, which is standard for multi-class classification tasks and well-suited for optimizing probabilistic predictions produced by neural networks.

Cross-entropy loss measures the discrepancy between the predicted class probabilities and the true class labels. Minimizing this loss encourages the model to assign high probability to the correct digit class for each input image.

### 6.2 Optimization Strategy

Model parameters are optimized using the Adam optimizer. Adam is an adaptive gradient-based optimization algorithm that combines the advantages of momentum-based methods and per-parameter learning rate adaptation. This makes it particularly effective for deep learning models, as it provides stable convergence and reduces the need for extensive manual learning rate tuning.

An initial learning rate of 1×10⁻³ was selected as a reasonable default based on common practice for CNNs trained on MNIST-like datasets. This choice provided a good balance between fast convergence and stable training behavior.

### 6.3 Training Loop and Epoch-Based Learning

Training is performed over multiple epochs, where each epoch corresponds to a full pass over the training dataset. During each epoch, the model processes mini-batches of images, computes the loss, performs backpropagation, and updates model parameters using gradient descent.

Batch-based training improves computational efficiency and allows the optimizer to generalize better by introducing stochasticity into the gradient updates. After each epoch, the model is evaluated on the test dataset to monitor generalization performance and detect potential overfitting.

Training metrics such as training loss, training accuracy, test loss, and test accuracy are recorded at each epoch. These metrics provide insight into the learning dynamics and allow for informed decisions regarding hyperparameter selection.

### 6.4 Hyperparameter Selection Strategy

Hyperparameter selection in this project was performed using informed manual selection rather than an automated or exhaustive search. Given the well-studied nature of the MNIST dataset and the relatively simple convolutional neural network architecture, standard hyperparameter values commonly reported in the literature were adopted as a starting point.

The learning rate was set to 1×10⁻³ when using the Adam optimizer, a value that is widely used for convolutional neural networks and known to provide stable convergence behavior. The batch size was set to 64, balancing computational efficiency with gradient stability. Other parameters, such as optimizer choice and loss function, were selected based on established best practices for multi-class image classification.

This approach allows the project to focus on the correctness and reproducibility of the pipeline while avoiding unnecessary computational overhead.

### 6.5 Empirical Observation During Training

Although multiple hyperparameter configurations were not exhaustively evaluated, training dynamics were closely monitored during model development. Metrics such as training loss, training accuracy, test loss, and test accuracy were recorded at each epoch to assess convergence behavior and generalization performance.

During training, the model exhibited rapid convergence within the first few epochs, followed by a gradual plateau in test accuracy. This behavior suggests that the chosen learning rate was sufficiently large to enable fast learning while remaining stable throughout training. No signs of training instability, such as exploding loss or erratic accuracy fluctuations, were observed.

The number of training epochs was selected based on these observations. Training beyond approximately 10 epochs yielded diminishing improvements in test performance, indicating that additional training provided limited benefit.

### 6.6 Overfitting Considerations

Overfitting was assessed by comparing training and test metrics across epochs. While training accuracy continued to increase slightly with additional epochs, test accuracy remained stable and did not degrade. Similarly, test loss did not exhibit a sustained upward trend.

This indicates that the model did not suffer from severe overfitting, likely due to the use of data augmentation, normalization, and a relatively compact model architecture. Based on these observations, the final model checkpoint was selected using the highest achieved test accuracy rather than the final training epoch.

6.7 Summary of Training and Hyperparameter Selection

In summary, hyperparameters were selected using standard values informed by prior knowledge and validated through empirical observation of training behavior. Monitoring of loss and accuracy curves ensured that the model converged stably and generalized well to unseen data. This approach provides a transparent and reproducible training procedure while remaining appropriate for the scope and objectives of the project.

## Training Procedure

## Evaluation

## Software Engineering Practices

## Deployment

## How to run the project locally

## Repository Structure

## References
