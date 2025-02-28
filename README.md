# AI Projects Repository

Welcome to my AI Projects Repository! This repository contains a collection of small AI projects I've worked on. Each project is designed to explore different aspects of artificial intelligence, machine learning, and data science. Feel free to explore, use, and contribute!

## Table of Contents

- [Project 1: Abnormal Behavior Detection using ML](https://github.com/renchuu/AI-ML-EDA/edit/main/README.md#project-1-abnormal-behavior-detection-using-ml)
- [Project 2: Fitness Pattern Analysis](https://github.com/renchuu/AI-ML-EDA/edit/main/README.md#project-2-fitness-pattern-analysis)
- [Project 3: Data Preprocessing for Criminal Records Dataset]()

## Project 1: Abnormal Behavior Detection using ML

### Description
This project aims to detect dangerous behaviors like smoking in gas stations or using a phone while driving using machine learning. It is based on the paper "Prediction of People’s Abnormal Behaviors Based on Machine Learning Algorithms" by Xiaoyu Song.

### Datasets
- **Cigarette Smoker Detection Dataset** from Kaggle
- **TIANCHI DATA SET** for calling people images
- **Person Face Dataset** from Kaggle for normal class

### Algorithms Used
- **Linear Support Vector Machine (LSVM)**
- **Kernel Support Vector Machine (KSVM)**
- **Decision Tree Classifier (DT)**
- **Random Forest Classifier (RF)**
- **K-Nearest Neighbors (KNN)**
- **K-Means Clustering**

## Project 2: Fitness Pattern Analysis

### Description
This project involves analyzing fitness patterns using various machine learning techniques. The goal is to explore different algorithms and methods to classify and predict fitness-related data effectively. The project includes data preprocessing, feature selection, and evaluation of different models. 

### Dataset
- **Gym Members Exercise Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)
  
### Data Preprocessing
1. **Data Loading**: Loaded labeled and unlabeled datasets.
2. **Feature Reduction**: Used PCA (Principal Component Analysis) to reduce dimensionality.
3. **Data Cleaning**: Removed missing values and irrelevant columns.
4. **Normalization**: Standardized the data to ensure uniformity.

### Algorithms Used
- **Logistic Regression**: Evaluated for classification tasks.
- **Support Vector Machine (SVM)**: Used for classification with both PCA-transformed and original data.
- **K-Nearest Neighbors (KNN)**: Tested with varying numbers of neighbors.
- **K-Means Clustering**: Applied for clustering analysis.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Used for visualization of high-dimensional data.

### Evaluation
- **Accuracy and Precision**: Measured the performance of classification algorithms.
- **Silhouette Score and Davies-Bouldin Index**: Evaluated the quality of clustering.


## Project 3: Data Preprocessing for Criminal Records Dataset

### Description
This project was my first attempt at data preprocessing, focusing on cleaning and preparing a dataset related to criminal records. The dataset is relatively small, which poses challenges for feature selection and model performance.

### Dataset
- **Crown Sentencing Dataset**: [Download from the Sentencing Council](https://www.sentencingcouncil.org.uk/research-and-resources/data-collections/crowncourt-sentencing-survey/)

### Data Preprocessing Steps
1. **Import Libraries**: Imported necessary libraries such as pandas, numpy, seaborn, and matplotlib.
2. **Load Data**: Loaded the dataset from a CSV file.
3. **Remove Duplicates**: Identified and removed duplicate entries.
4. **Handle Missing Values**: Replaced missing values with appropriate placeholders (e.g., mean for numerical columns, mode for categorical columns).
5. **Remove Irrelevant Features**: Dropped columns that were not relevant to the analysis.
6. **Feature Selection**: Attempted to select relevant features, but due to the small dataset size, the results were not optimal.
7. **Normalization**: Standardized numerical features to ensure uniformity.
8. **Visualization**: Created visualizations to understand the distribution and relationships within the data.

### Challenges
- **Small Dataset**: The dataset size was small, making feature selection less effective.
- **Highly Correlated Features**: Identified and removed highly correlated features to reduce redundancy.
- **Irrelevant Features**: Removed features that did not contribute significantly to the analysis.

### Model Training
- **Random Forest Classifier**: Used as the primary model for classification tasks.
- **Train-Test Split**: Split the data into training and testing sets.
- **Model Evaluation**: Evaluated the model using accuracy and classification reports.
