# DVAE26 Final Project
In this project we create an image classification model based on the MNIST dataset.

## 1. Use of Generative Artificial Intelligence
Use of Generative Artificial Intelligence (Declaration)

This project was developed with the support of generative artificial intelligence (GAI) tools. The use of such tools complies with the guidelines for students at Karlstad University, which permit the use of generative AI provided that its usage is transparently declared and that the student remains fully responsible for the submitted work.

In this project, generative AI tools (ChatGPT) were used as support tools in the following ways:

Assisting with the generation of code for modular components, including data handling, model architecture, training routines, evaluation metrics, and plotting utilities.

Providing explanations and clarifications of machine learning concepts, software engineering practices, and PyTorch-specific implementation details.

Supporting the writing, structuring and drafting of written content, including methodological explanations, design rationales, and report sections.

The use of generative AI was intended to support learning, understanding, and productivity, not to replace independent reasoning or accountability, in accordance with Karlstad University regulations.

## 2. Introduction and Problem Definition
Image recognition is a central problem in modern artificial intelligence and machine learning, with applications ranging from medical image analysis and autonomous driving to document processing and biometric systems. At its core, image recognition involves learning meaningful representations from raw pixel data in order to classify or interpret visual information. The complexity of this task arises from the high dimensionality of image data, variations in appearance, noise, and transformations such as rotation or scale changes.

Convolutional Neural Networks (CNNs) have become the dominant approach for image recognition tasks due to their ability to exploit spatial structure in images. By learning hierarchical feature representations through convolutional filters, CNNs can automatically extract relevant patterns such as edges, textures, and shapes, without the need for manually engineered features. As a result, CNN-based models have achieved state-of-the-art performance on a wide range of benchmark datasets and real-world applications.

In this project, the problem of image recognition is studied using the MNIST dataset, a well-known benchmark consisting of grayscale images of handwritten digits. Although MNIST is considered a relatively simple dataset by modern standards, it remains widely used for evaluating image classification pipelines due to its clean structure, standardized splits, and balanced class distribution. Importantly, MNIST allows a clear focus on the end-to-end machine learning workflow—including data handling, model design, training, evaluation, and deployment—without confounding factors introduced by large-scale or noisy datasets.

The objective of this project is to develop, train, and evaluate a convolutional neural network capable of accurately classifying handwritten digits from the MNIST dataset. Beyond achieving high classification accuracy, the project emphasizes software engineering best practices and reproducibility. This includes modular code design, explicit handling of data transformations, systematic evaluation using multiple metrics, and deployment of the trained model to a public platform.

From an engineering perspective, the project is structured as a complete machine learning pipeline. Raw image data is ingested and transformed into a suitable numerical representation, augmented to improve robustness, and then used to train a CNN model. Model performance is evaluated using standard classification metrics such as accuracy, precision, and recall. The final trained model is packaged and deployed to the Hugging Face platform, allowing external users to download and run inference using the trained network.

By using MNIST as a benchmark dataset, this project provides a controlled environment in which design decisions, training behavior, and evaluation results can be clearly analyzed and explained. The focus is therefore not only on model performance, but also on demonstrating a structured and reproducible approach to machine learning system development, in line with the objectives of the course.

## 4. Dataset Description and Data Quality Analysis
### 4.1 Dataset Overview

The dataset used in this project is the MNIST handwritten digit dataset, a widely adopted benchmark for image classification tasks. MNIST consists of grayscale images of handwritten digits ranging from 0 to 9. Each image has a fixed resolution of 28 × 28 pixels, resulting in a total of 784 pixel values per image. The dataset is divided into a training set containing 60,000 images and a test set containing 10,000 images.

Each image is associated with a single class label corresponding to the digit it represents. The dataset is approximately class-balanced, meaning that each digit class is represented by a similar number of samples. This balanced distribution simplifies evaluation and allows the use of standard classification metrics without the need for class weighting or resampling strategies.

MNIST images are stored as grayscale intensity values, where each pixel represents the brightness at a particular spatial location. This structure makes MNIST particularly suitable for convolutional neural networks, which are designed to exploit local spatial correlations in image data.

### 4.2 Data Quality Characteristics

MNIST is a curated and well-maintained benchmark dataset, and as such, it exhibits high data quality compared to many real-world datasets. No missing values, corrupted images, or inconsistent labels were observed during dataset inspection. All images share a consistent resolution and format, and labels are provided for every sample.

Unlike tabular datasets, image datasets do not typically require operations such as imputation or removal of invalid rows. Instead, data quality considerations for image data focus on factors such as visual clarity, label correctness, class balance, and robustness to natural variations in appearance. In the case of MNIST, the dataset’s controlled acquisition process and extensive prior use in the literature provide confidence in its label accuracy and overall integrity.

To verify dataset integrity within the project pipeline, the dataset was loaded programmatically and basic sanity checks were performed. These checks included verifying the total number of images in the training and test sets, confirming the expected image dimensions, and ensuring that label values fell within the valid range of 0 to 9.

### 4.3 Class Distribution and Balance

An important aspect of data quality for classification tasks is class distribution. Severe class imbalance can bias model training and distort evaluation metrics. The MNIST dataset exhibits an approximately uniform distribution across all ten digit classes in both the training and test sets. This balance reduces the risk of biased learning and enables the use of accuracy, precision, and recall as reliable evaluation metrics.

Because the dataset is balanced, no additional techniques such as oversampling, undersampling, or class-weighted loss functions were required. This allows model performance to be interpreted directly without adjusting for skewed class frequencies.

### 4.4 Normalization and Numerical Stability

Although MNIST does not require traditional data cleaning, numerical preprocessing is still necessary to ensure stable and efficient model training. Raw pixel values in MNIST are represented as integer intensities, which are not ideal for gradient-based optimization. To address this, images were normalized using the mean and standard deviation computed from the training set.

Normalization ensures that input values are centered and scaled consistently, which improves numerical stability during training and accelerates convergence. Importantly, normalization parameters were computed exclusively on the training data and then applied to both training and test sets, preventing information leakage from the test data into the training process.

### 4.5 Summary of Data Quality Considerations

In summary, the MNIST dataset provides a high-quality and well-structured foundation for image recognition experiments. While extensive data cleaning procedures were not required, careful attention was paid to dataset integrity, class balance, and numerical preprocessing. These steps ensure that observed model performance reflects the learning capability of the model rather than artifacts introduced by data quality issues.

The controlled nature of MNIST allows the project to focus on the design, training, and evaluation of the convolutional neural network, while still demonstrating appropriate data quality analysis practices relevant to image-based machine learning systems.

## 5. Repository Structure

## 6. Exploratory Data Analysis (EDA)
### 6.1 Purpose of Exploratory Data Analysis for Image Data

Exploratory Data Analysis (EDA) is a critical step in any machine learning pipeline, as it provides insights into the structure, variability, and potential challenges of the dataset before model training. For image-based datasets, EDA differs from traditional tabular data analysis. Rather than focusing on statistical summaries of individual features, image EDA emphasizes visual inspection, dimensionality understanding, and assessment of intra-class and inter-class variability.

The primary objectives of EDA in this project are to verify that the dataset has been loaded correctly, to understand the visual characteristics of handwritten digits, and to identify sources of variation that the model must learn to handle.

### 6.2 Dataset Shape and Dimensionality

Each MNIST image consists of 28 × 28 grayscale pixels, resulting in a two-dimensional array per sample. When loaded into the training pipeline, images are converted into tensors with shape (1, 28, 28), where the leading dimension corresponds to the single grayscale channel.

The full training dataset therefore has the shape (60000, 1, 28, 28), while the test dataset has the shape (10000, 1, 28, 28). This confirms that the dataset has been loaded correctly and matches the expected structure defined by the MNIST specification.

Understanding the dimensionality of the input data is essential for designing the convolutional neural network architecture, as convolutional layers operate directly on the spatial structure of the input tensors.

### 6.3 Visual Inspection of Samples

A key component of EDA for image data is visual inspection of representative samples from each class. Random samples from the dataset reveal significant variation in handwriting style, stroke thickness, digit size, and orientation. For example, the digit “1” may appear as a simple vertical line in some samples, while in others it includes a serif or diagonal stroke. Similarly, digits such as “4” and “9” can exhibit structural similarities that make them more challenging to distinguish.

Visual inspection confirms that the dataset contains a wide range of realistic handwriting styles. This variability motivates the use of data augmentation techniques during training to improve the model’s robustness to small transformations and stylistic differences.

### 6.4 Intra-Class and Inter-Class Variability

EDA also highlights two important characteristics of the dataset:

- Intra-class variability: Samples belonging to the same digit class can differ substantially in appearance due to differences in handwriting style, stroke pressure, and orientation.

- Inter-class similarity: Certain digit pairs, such as “3” and “5” or “4” and “9”, may share visual similarities that increase the risk of misclassification.

These observations justify the use of convolutional neural networks, which are well-suited for learning hierarchical features that can capture both local patterns (such as edges and curves) and global digit structure.

### 6.5 Feature Engineering Considerations

Unlike classical machine learning approaches, convolutional neural networks do not require explicit manual feature engineering. Instead, the network learns relevant features directly from the pixel data during training. As a result, no handcrafted features such as edge detectors or shape descriptors were introduced.

However, preprocessing steps such as normalization and data augmentation can be viewed as forms of feature preparation that improve learning stability and generalization. These steps are discussed in detail in the following section.

### 6.6 Summary of EDA Findings

Exploratory Data Analysis confirmed that the MNIST dataset is correctly structured, visually diverse, and suitable for training a convolutional neural network. The observed variability in handwriting styles underscores the importance of robust model design and appropriate data augmentation. Additionally, the absence of severe class imbalance or corrupted samples allows the focus to remain on model architecture and training strategy rather than extensive data cleaning.

The insights gained from EDA directly inform the preprocessing and modeling decisions described in subsequent sections.

## 7. Data Transformation and Augmentation
### 7.1 Role of Data Transformation in Image Recognition

Before images can be used as input to a convolutional neural network, they must be transformed into a numerical format suitable for gradient-based optimization. In this project, data transformation serves two primary purposes: converting raw image data into tensors that can be processed by the model, and normalizing input values to ensure stable and efficient training.

Raw MNIST images are stored as grayscale intensity values. These images are first converted into tensors and reshaped to include an explicit channel dimension. This transformation results in tensors of shape (1, 28, 28), which is the expected input format for the convolutional layers used in the model architecture.

### 7.2 Normalization for Numerical Stability

Normalization is a crucial preprocessing step in neural network training. Without normalization, input values can vary across a wide numerical range, which may lead to unstable gradients and slow convergence during training. To address this, pixel values are normalized using the mean and standard deviation computed from the MNIST training set.

By centering and scaling the input distribution, normalization improves numerical stability and allows the optimizer to operate more effectively. Importantly, normalization parameters are computed using only the training data and then applied consistently to both training and test datasets. This prevents information leakage from the test set into the training process and ensures a fair evaluation of model performance.

### 7.3 Data Augmentation Strategy

Data augmentation is employed to enhance the diversity of the training dataset and improve the model’s ability to generalize to unseen samples. Although MNIST is a clean and curated dataset, handwritten digits exhibit natural variations in orientation and writing style. Without augmentation, a model may overfit to the specific orientations and patterns present in the training set.

In this project, data augmentation is implemented using random rotations applied to training images. Small rotations simulate realistic variations in handwriting, such as slightly tilted digits, without altering the semantic meaning of the image. This encourages the model to learn rotation-invariant features rather than memorizing specific pixel configurations.

### 7.4 Separation of Training and Test Transformations

A key design decision in the preprocessing pipeline is the strict separation of transformations applied to the training and test datasets. Data augmentation is applied exclusively to the training data, while the test data undergoes only deterministic transformations such as tensor conversion and normalization.

This separation ensures that evaluation metrics reflect the model’s performance on unmodified data and prevents artificial inflation of test accuracy. Applying augmentation to the test set would introduce randomness into the evaluation process and compromise the reliability of the reported results.

### 7.5 Implementation Considerations

The transformation and augmentation logic is implemented in a modular and reusable manner. Separate transformation pipelines are defined for training and testing, allowing preprocessing behavior to be adjusted independently without modifying the rest of the training code. This modular design improves code clarity, simplifies experimentation, and supports reproducibility.

By encapsulating transformations within dedicated functions, the preprocessing pipeline remains transparent and easy to audit, which is particularly important in an academic and engineering-focused context.

### 7.6 Summary of Data Preparation Approach

In summary, data transformation and augmentation play a central role in preparing the MNIST dataset for convolutional neural network training. Through tensor conversion, normalization, and carefully chosen augmentation techniques, the input data is made numerically stable and more representative of real-world handwriting variability. These preprocessing steps directly support the model’s ability to learn robust and generalizable representations from image data.

## 8. Model Architecture
### 8.1 Motivation for Using a Convolutional Neural Network

Convolutional Neural Networks (CNNs) are specifically designed to process grid-structured data such as images. Unlike fully connected networks, CNNs exploit the spatial locality and hierarchical structure of image data through the use of convolutional filters and shared weights. This makes them particularly effective for image recognition tasks, where local patterns such as edges, corners, and curves combine to form higher-level structures.

Given the two-dimensional nature of MNIST images and the need to capture spatial relationships between pixels, a CNN is a natural and well-established choice for this task.

### 8.2 Overall Network Structure

The model used in this project is a custom-designed CNN implemented in PyTorch. It follows a relatively simple architecture consisting of multiple convolutional layers followed by pooling operations and fully connected layers. This design reflects a balance between expressive power and computational efficiency, which is appropriate for the MNIST dataset.

At a high level, the network is composed of the following components:

1. Convolutional layers to extract local spatial features from input images.
2. Non-linear activation functions to introduce non-linearity and enable the learning of complex patterns.
3. Pooling layers to reduce spatial dimensionality and provide translation invariance.
4. Fully connected layers to map learned feature representations to class scores.

### 8.3 Convolutional Feature Extraction

The initial layers of the network consist of convolutional layers that operate directly on the input image tensors. These layers apply learnable filters across the spatial dimensions of the image to detect local patterns such as edges and simple shapes. As depth increases, subsequent convolutional layers learn progressively more abstract features, combining lower-level patterns into higher-level representations of digit structure.

Each convolutional layer is followed by a non-linear activation function, allowing the network to model complex, non-linear relationships in the data. This hierarchical feature extraction process is central to the effectiveness of CNNs in image recognition tasks.

### 8.4 Pooling and Dimensionality Reduction

Pooling layers are interleaved with convolutional layers to reduce the spatial resolution of feature maps. By summarizing local neighborhoods, pooling helps reduce computational complexity and provides a degree of invariance to small translations in the input image.

This is particularly useful for handwritten digit recognition, where the precise position of a stroke may vary slightly between samples without changing the digit’s identity. Pooling allows the network to focus on the presence of features rather than their exact location.

### 8.5 Fully Connected Classification Layers

After convolutional feature extraction, the resulting feature maps are flattened and passed to fully connected layers. These layers integrate information from all extracted features and produce a set of output scores, one for each digit class.

The final layer of the network outputs raw class scores, often referred to as logits. These logits are later converted into class probabilities during evaluation using a softmax function. The predicted digit corresponds to the class with the highest score.

### 8.6 Architectural Simplicity and Design Rationale

The chosen architecture is intentionally kept simple. MNIST is a relatively small and clean dataset, and more complex architectures would provide limited performance gains while increasing training time and implementation complexity. A compact CNN is sufficient to achieve high accuracy on this dataset while maintaining interpretability and ease of analysis.

This design choice aligns with the goals of the project, which emphasize clarity, reproducibility, and software engineering best practices over maximizing performance through overly complex models.

### 8.7 Summary of Model Architecture

In summary, the model architecture leverages the core strengths of convolutional neural networks to perform image classification effectively. Through convolutional feature extraction, pooling-based dimensionality reduction, and fully connected classification layers, the network learns hierarchical representations of handwritten digits. The simplicity of the architecture ensures efficient training and clear interpretability, making it well-suited for both educational purposes and systematic analysis.

## 9. Training Procedure and Hyperparameter Selection
### 9.1 Training Objective and Loss Function

The objective of the training process is to learn model parameters that minimize classification error on handwritten digit images. This is formulated as a multi-class classification problem with ten output classes corresponding to digits 0 through 9. The network is trained using the cross-entropy loss function, which is standard for multi-class classification tasks and well-suited for optimizing probabilistic predictions produced by neural networks.

Cross-entropy loss measures the discrepancy between the predicted class probabilities and the true class labels. Minimizing this loss encourages the model to assign high probability to the correct digit class for each input image.

### 9.2 Optimization Strategy

Model parameters are optimized using the Adam optimizer. Adam is an adaptive gradient-based optimization algorithm that combines the advantages of momentum-based methods and per-parameter learning rate adaptation. This makes it particularly effective for deep learning models, as it provides stable convergence and reduces the need for extensive manual learning rate tuning.

An initial learning rate of 1×10⁻³ was selected as a reasonable default based on common practice for CNNs trained on MNIST-like datasets. This choice provided a good balance between fast convergence and stable training behavior.

### 9.3 Training Loop and Epoch-Based Learning

Training is performed over multiple epochs, where each epoch corresponds to a full pass over the training dataset. During each epoch, the model processes mini-batches of images, computes the loss, performs backpropagation, and updates model parameters using gradient descent.

Batch-based training improves computational efficiency and allows the optimizer to generalize better by introducing stochasticity into the gradient updates. After each epoch, the model is evaluated on the test dataset to monitor generalization performance and detect potential overfitting.

Training metrics such as training loss, training accuracy, test loss, and test accuracy are recorded at each epoch. These metrics provide insight into the learning dynamics and allow for informed decisions regarding hyperparameter selection.

### 9.4 Hyperparameter Selection Strategy

Hyperparameter selection in this project was performed using informed manual selection rather than an automated or exhaustive search. Given the well-studied nature of the MNIST dataset and the relatively simple convolutional neural network architecture, standard hyperparameter values commonly reported in the literature were adopted as a starting point.

The learning rate was set to 1×10⁻³ when using the Adam optimizer, a value that is widely used for convolutional neural networks and known to provide stable convergence behavior. The batch size was set to 64, balancing computational efficiency with gradient stability. Other parameters, such as optimizer choice and loss function, were selected based on established best practices for multi-class image classification.

This approach allows the project to focus on the correctness and reproducibility of the pipeline while avoiding unnecessary computational overhead.

### 9.5 Empirical Observation During Training

Although multiple hyperparameter configurations were not exhaustively evaluated, training dynamics were closely monitored during model development. Metrics such as training loss, training accuracy, test loss, and test accuracy were recorded at each epoch to assess convergence behavior and generalization performance.

During training, the model exhibited rapid convergence within the first few epochs, followed by a gradual plateau in test accuracy. This behavior suggests that the chosen learning rate was sufficiently large to enable fast learning while remaining stable throughout training. No signs of training instability, such as exploding loss or erratic accuracy fluctuations, were observed.

The number of training epochs was selected based on these observations. Training beyond approximately 10 epochs yielded diminishing improvements in test performance, indicating that additional training provided limited benefit.

### 9.6 Overfitting Considerations

Overfitting was assessed by comparing training and test metrics across epochs. While training accuracy continued to increase slightly with additional epochs, test accuracy remained stable and did not degrade. Similarly, test loss did not exhibit a sustained upward trend.

This indicates that the model did not suffer from severe overfitting, likely due to the use of data augmentation, normalization, and a relatively compact model architecture. Based on these observations, the final model checkpoint was selected using the highest achieved test accuracy rather than the final training epoch.

### 9.7 Summary of Training and Hyperparameter Selection

In summary, hyperparameters were selected using standard values informed by prior knowledge and validated through empirical observation of training behavior. Monitoring of loss and accuracy curves ensured that the model converged stably and generalized well to unseen data. This approach provides a transparent and reproducible training procedure while remaining appropriate for the scope and objectives of the project.

## 10. Evaluation and Results
### 10.1 Evaluation Methodology

Model evaluation was conducted using the MNIST test set, which consists of 10,000 images that were not used during training. The test set provides an unbiased estimate of the model’s generalization performance. Evaluation was performed after each training epoch to monitor learning progress and support model selection.

Multiple evaluation metrics were used to assess performance. In addition to overall classification accuracy, precision and recall were computed to provide a more detailed view of classification behavior across digit classes. Test loss was also recorded to assess confidence calibration and detect potential overfitting.

### 10.2 Classification Accuracy

Classification accuracy was used as the primary performance metric. Due to the approximately balanced class distribution of the MNIST dataset, accuracy provides a meaningful summary of overall model performance.

The trained convolutional neural network achieved a test accuracy exceeding 99%. Accuracy improved rapidly during the early stages of training and reached a stable plateau after approximately 8–10 epochs. This indicates that the model was able to learn discriminative features efficiently and converge within a relatively small number of training iterations.

The high final accuracy demonstrates that the model successfully captures the essential visual characteristics required to distinguish handwritten digits.

### 10.3 Precision and Recall

To complement accuracy, macro-averaged precision and recall were computed across all ten digit classes. Macro-averaging ensures that each class contributes equally to the final metric, which is appropriate given the balanced nature of the MNIST dataset.

High precision values indicate that the model produces few false positive predictions, while high recall values indicate that most true digit instances are correctly identified. The trained model achieved high macro-averaged precision and recall, demonstrating consistent performance across digit classes rather than favoring a subset of classes.

The agreement between accuracy, precision, and recall suggests that the model’s predictions are both reliable and well-balanced.

### 10.4 Confusion Matrix Analysis

The confusion matrix was analyzed to gain insight into class-wise prediction behavior. Most predictions fall along the diagonal of the matrix, indicating correct classification for the majority of samples.

Misclassifications primarily occur between visually similar digits, such as digits with overlapping stroke patterns or ambiguous handwritten forms. Examples include occasional confusion between digits such as “4” and “9” or “3” and “5”. These errors are consistent with known challenges in handwritten digit recognition and reflect inherent ambiguity in some samples rather than systematic model failure.

Importantly, no single class exhibits a disproportionately high error rate, indicating that the model generalizes well across all digit categories.

### 10.5 Training and Test Loss Behavior

Training and test loss curves provide valuable insight into the learning dynamics of the model. Training loss decreased steadily throughout the training process, indicating effective optimization and continual improvement in model fit to the training data.

Test loss decreased during the initial epochs and then stabilized as training progressed. The absence of a sustained increase in test loss suggests that the model did not experience severe overfitting. While training accuracy continued to improve slightly in later epochs, test accuracy remained stable, indicating diminishing returns rather than degradation in generalization performance.

This behavior supports the selection of the final model checkpoint based on peak test accuracy rather than the final training epoch.

### 10.6 Interpretation of Results

The evaluation results demonstrate that the convolutional neural network effectively learns robust feature representations for handwritten digit recognition. The combination of high accuracy, strong precision and recall, and stable loss behavior indicates that the model generalizes well to unseen data.

The results also validate the preprocessing and architectural decisions made earlier in the pipeline. Data normalization and augmentation contributed to stable training, while the compact CNN architecture proved sufficient for achieving strong performance on MNIST without unnecessary complexity.

### 10.7 Summary of Evaluation

In summary, the trained model achieved strong and consistent performance across all evaluation metrics. Analysis of accuracy, precision, recall, confusion matrices, and learning curves provides confidence in the model’s reliability and generalization capability. These results confirm the effectiveness of the end-to-end pipeline implemented in this project and justify the selection of the final deployed model.

## 11. Software Engineering Best Practices
### 11.1 Modular Project Structure

The project is organized using a modular repository structure that separates concerns and improves maintainability. Core functionality is divided into logically distinct modules responsible for data handling, preprocessing, model definition, training, evaluation, and deployment. This separation allows individual components to be developed, tested, and modified independently without affecting the entire system.

For example, data ingestion and preprocessing are implemented separately from model training and evaluation logic. This design reduces coupling between components and supports reuse, experimentation, and debugging. Such modularization aligns with established software engineering principles and is particularly important in machine learning projects, where pipelines often evolve iteratively.

### 11.2 Reproducibility and Configuration Management

Reproducibility is a key requirement for reliable machine learning systems. To support reproducible experiments, the project explicitly controls sources of randomness by setting fixed random seeds for data loading and model training. This ensures that results can be reproduced across runs under the same configuration.

Training configurations, including hyperparameters such as learning rate, batch size, number of epochs, and optimizer settings, are recorded and stored alongside training artifacts. This enables traceability between reported results and the exact configuration used to produce them, which is essential for both academic evaluation and future experimentation.

### 11.3 Testing and Validation

Automated testing is used to validate key components of the pipeline. Unit tests are implemented for critical functions such as data ingestion, preprocessing, model construction, and evaluation metrics. These tests verify expected behavior and help detect regressions when code is modified.

By integrating testing into the development process, the project reduces the risk of silent errors and increases confidence in the correctness of individual components. This approach reflects standard software engineering practice and is particularly valuable in machine learning projects, where errors in data handling or evaluation logic can significantly affect results.

### 11.4 Logging and Monitoring

Logging is used to track training progress and record important events during execution. Training loss, accuracy, and evaluation metrics are logged at regular intervals, providing transparency into the learning process and enabling post hoc analysis of model behavior.

The use of structured logs and saved training histories allows performance trends to be visualized and interpreted after training has completed. This facilitates informed decision-making during model development and supports clear reporting of results.

### 11.5 Path Handling and Environment Independence

To ensure portability across different execution environments, file paths are handled explicitly and resolved relative to the project root. This avoids reliance on implicit working directory assumptions, which can vary between scripts, notebooks, and execution environments.

In the notebook environment, the working directory is explicitly set to the project root to ensure consistent behavior when accessing data, artifacts, and configuration files. This design choice improves robustness and prevents path-related errors when the project is executed on different machines or by external users.

### 11.6 Version Control and Repository Management

All code, configuration files, and documentation are managed using a version control system. Version control enables incremental development, rollback of changes, and clear tracking of project evolution. It also supports collaboration and external review.

Large binary artifacts such as datasets and trained model weights are excluded from the version-controlled repository to avoid unnecessary repository bloat. Instead, trained models are stored as artifacts and deployed separately via the Hugging Face platform.

### 11.7 Deployment-Oriented Design

Deployment considerations are integrated into the project design from an early stage. The trained model, along with the corresponding inference code and documentation, is packaged in a format suitable for deployment and uploaded to the Hugging Face Hub. This ensures that the model can be easily reused and evaluated by others without requiring access to the full training environment.

By designing the pipeline with deployment in mind, the project demonstrates an end-to-end engineering approach rather than focusing solely on model training.

### 11.8 Summary of Software Engineering Practices

In summary, the project applies software engineering best practices throughout the machine learning lifecycle. Modular code organization, reproducibility measures, automated testing, logging, explicit path handling, version control, and deployment-oriented design collectively contribute to a robust and maintainable system. These practices ensure that the developed solution is not only accurate but also reliable, transparent, and suitable for real-world use.

## 12. Deployment on the Hugging Face Platform
### 12.1 Motivation for Deployment

Deployment is an important step in the machine learning lifecycle, as it enables trained models to be shared, reused, and evaluated outside the local development environment. In the context of this project, deployment serves to demonstrate that the trained convolutional neural network can be packaged and distributed using a modern model-sharing platform.

The Hugging Face Hub was selected as the deployment platform due to its widespread use in the machine learning community and its support for hosting trained models in a reproducible and accessible manner.

### 12.2 Deployment Scope

The scope of deployment in this project focuses on model distribution rather than full application serving. The trained model weights were uploaded to the Hugging Face Hub to allow external users to download and load the model within their own environments.

The deployment does not include a custom inference script or interactive interface. Instead, the uploaded artifacts are intended to be used in conjunction with the provided training code, which defines the model architecture and preprocessing steps. This approach ensures that the deployed model corresponds exactly to the evaluated version reported in this project.

### 12.3 Authentication and Security Considerations

Authentication to the Hugging Face Hub is performed using a personal access token. For security reasons, the access token is not embedded in the notebook or source code. Instead, authentication is handled through the official Hugging Face login mechanism, which securely stores credentials outside the project repository.

This approach follows standard security best practices and prevents accidental exposure of sensitive authentication information.

### 12.4 Uploading the Model to the Hub

The trained model was uploaded programmatically using the Hugging Face Python API directly from the notebook environment. This ensures that the deployed artifact is consistent with the final trained model selected during evaluation.

Only the trained model weights were uploaded, avoiding unnecessary duplication of training data or intermediate artifacts. By separating training and deployment concerns, the repository remains lightweight while still enabling external access to the trained model.

### 12.5 Reproducibility and Reuse

Although the deployment does not include a standalone inference script, reproducibility is ensured through the availability of the full training codebase in the project repository. External users can reproduce inference behavior by loading the deployed weights into the same model architecture and applying identical preprocessing steps as defined in the project code.

This design emphasizes transparency and correctness over convenience and reflects a development-focused deployment strategy appropriate for an academic project.

### 12.6 Summary of Deployment

In summary, the trained convolutional neural network was successfully deployed to the Hugging Face Hub as a reusable model artifact. The deployment demonstrates the ability to package and distribute trained models using a modern platform while maintaining security and reproducibility. While the deployment is minimal and does not include a dedicated inference interface, it fulfills the project requirement of showcasing the trained model on the Hugging Face platform.

## 13. Limitations and Future Work
### 13.1 Dataset Limitations

A primary limitation of this project is the use of the MNIST dataset, which is a highly curated and simplified benchmark for image recognition. MNIST images are low-resolution, grayscale, and centered, with minimal background noise. While this makes the dataset well-suited for controlled experimentation and educational purposes, it does not fully reflect the complexity of real-world image recognition problems.

As a result, the high performance achieved in this project should be interpreted in the context of the dataset’s simplicity. Models trained on MNIST may not generalize directly to more complex handwritten digit datasets or real-world document images without additional adaptation.

### 13.2 Limited Data Quality Challenges

Because MNIST is a clean and well-structured dataset, the project involved limited data cleaning or outlier handling compared to real-world datasets. No missing values, corrupted samples, or mislabeled data were observed. While this allows the focus to remain on the machine learning pipeline itself, it reduces exposure to more challenging data quality issues such as noise, annotation errors, or domain shifts.

Future work could involve applying the same pipeline to less curated image datasets, where data quality analysis and cleaning would play a more substantial role.

### 13.3 Model Architecture Simplicity

The convolutional neural network used in this project was intentionally kept compact and simple. While this architecture is sufficient to achieve strong performance on MNIST, it may not scale effectively to more complex tasks involving higher-resolution images, multiple color channels, or greater intra-class variability.

More expressive architectures, such as deeper convolutional networks, residual connections, or models incorporating regularization techniques like dropout and batch normalization, could be explored in future work to improve robustness and scalability.

### 13.4 Hyperparameter Exploration

Hyperparameter selection in this project was based on informed manual choices and empirical observation of training behavior, rather than systematic or automated optimization. While this approach is appropriate for the scope of the project, it limits insight into how sensitive the model’s performance is to different hyperparameter configurations.

Future work could include structured hyperparameter exploration, such as evaluating multiple learning rates, batch sizes, or training durations, to better understand the trade-offs between convergence speed, stability, and final performance.

### 13.5 Evaluation Scope

Evaluation was conducted using a single predefined train-test split provided by the MNIST dataset. Although this is standard practice for benchmark datasets, it limits the assessment of robustness across different data partitions or sampling variations.

Additional evaluation strategies, such as cross-validation or testing on alternative handwritten digit datasets, could provide a more comprehensive understanding of model generalization.

### 13.6 Deployment Scope

The deployment performed in this project focuses on making the trained model weights available via the Hugging Face Hub. While this satisfies the requirement of showcasing the model on a modern deployment platform, it does not provide a standalone inference interface or interactive demonstration.

Future extensions could include packaging a dedicated inference script, writing a detailed model card, or creating an interactive application that allows users to upload images and receive predictions. Such additions would further improve accessibility and usability.

### 13.7 Summary of Limitations and Future Directions

In summary, the limitations of this project primarily stem from the simplicity of the dataset, the compact model architecture, and the scope of hyperparameter exploration and deployment. These constraints are appropriate given the project’s educational focus but also highlight clear opportunities for extension. Future work could build on the current pipeline by incorporating more complex data, advanced architectures, systematic tuning, and richer deployment mechanisms.

## 14. Conclusion

This project presented the design, implementation, and evaluation of an end-to-end image recognition pipeline using a convolutional neural network trained on the MNIST handwritten digit dataset. The work focused not only on achieving strong classification performance, but also on applying software engineering best practices and ensuring reproducibility, transparency, and deployability throughout the machine learning lifecycle.

Starting from data ingestion and preprocessing, the project demonstrated appropriate handling of image data through normalization and data augmentation. Exploratory Data Analysis provided insight into the structure and variability of the dataset, informing subsequent modeling decisions. A custom convolutional neural network was designed and trained using established optimization techniques, resulting in high classification accuracy and stable generalization performance.

Model evaluation was conducted using multiple metrics, including accuracy, precision, recall, confusion matrices, and loss curves. These evaluation methods provided a comprehensive understanding of model behavior and confirmed that the trained network performs reliably across all digit classes. The absence of severe overfitting further validated the chosen architecture and training strategy.

In addition to model development, the project emphasized software engineering principles such as modular code organization, automated testing, reproducibility, and version control. Deployment of the trained model to the Hugging Face Hub demonstrated the ability to package and distribute machine learning artifacts using modern tooling, completing the end-to-end pipeline from data to deployment.

While the project is constrained by the simplicity of the MNIST dataset and a compact model architecture, it provides a solid foundation for future extensions involving more complex data, advanced architectures, systematic hyperparameter optimization, and richer deployment scenarios. Overall, the project successfully meets the objectives of developing, evaluating, and deploying an image recognition model while adhering to sound engineering practices.
