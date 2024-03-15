# MovieSent

## Movie Review Sentiment Analysis using Deep Learning

# Movie Review Sentiment Analysis using Deep Learning

[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.9-orange)](https://www.tensorflow.org/)



## Overview
This project focuses on sentiment analysis of movie reviews using deep learning techniques. The goal is to classify movie reviews as positive or negative based on the sentiment expressed in the text. The analysis involves data preprocessing steps including normalization, stop words removal, and lemmatization, followed by the use of embedding techniques and LSTM (Long Short-Term Memory) neural networks for classification.

## Dataset
You can find the dataset used in this project on Kaggle. Click [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Key Terms and Methodologies

### Data Preprocessing
Data preprocessing involves cleaning and transforming raw data into a format suitable for analysis. In this project, the following techniques are used:

- **Normalization:** Converting text to lowercase and removing non-alphanumeric characters to ensure consistency in the text data.
- **Stop Words Removal:** Eliminating common words (such as "the," "is," "and") that do not contribute much to the sentiment analysis.
- **Lemmatization:** Reducing words to their base or root form to simplify the text and improve the accuracy of analysis.

These techniques are applied to enhance the quality of the input data for the sentiment analysis model.

### Embedding
In this project, an Embedding is a technique used to represent words as dense vectors in a continuous vector space.This embedding layer converts input text into fixed-size dense vectors, capturing semantic similarities between words.

### LSTM (Long Short-Term Memory)
In this project, LSTM networks are used to capture long-term dependencies in sequential data. This layers can learn patterns and relationships within movie reviews, enabling the model to make accurate sentiment predictions.

## Model Architecture
The deep learning model architecture consists of the following layers:

1. **Embedding Layer:** Converts input text into dense vectors of fixed size (`embedded_feature_size`) to represent words.
2. **LSTM Layer:** Long Short-Term Memory layer with 128 units to learn patterns in sequential data.
3. **Dropout Layer:** Regularization technique to prevent overfitting by randomly dropping neurons during training.
4. **Batch Normalization:** Normalizes the activations of the previous layer for faster training and improved stability.
5. **Dense Layer:** Output layer with sigmoid activation function for binary classification (positive/negative sentiment).

### Techniques Used in Model

- **Dropout:** Dropout layer is utilized with a rate of 0.6 in this project. It serves as a regularization technique to prevent overfitting during model training, thus improving the model's generalization ability.
  
- **L2 Regularization:** L2 regularization is applied with a regularization strength (lambda) of 0.01 to penalise the loss function preventing the model from fitting.

- **Batch Normalization:** This project uses default parameters for batch normalization reducing internal covariate shift and enabling higher learning rates, further enhances training speed and stability.

These techniques are incorporated into the model architecture to improve its robustness, generalization ability, and training efficiency.

## Data Preprocessing

- Loading the dataset (`IMDB_Dataset.csv`).
- Visualizing the distribution of sentiment labels in the dataset.
- Removing patterns such as URLs, user mentions, punctuations, repeating characters, and break words from the reviews.
- Applying stop words removal and lemmatization using spaCy.

## Libraries Used
- **Data Preprocessing:** pandas, numpy, re
- **Deep Learning:** TensorFlow (Keras API), EarlyStopping, Regularizers
- **NLP:** spaCy
- **Evaluation Metrics:** scikit-learn (classification_report, accuracy_score, confusion_matrix), train_test_split
- **Visualization:** matplotlib, seaborn, WordCloud


## Training and Evaluation
The model is trained using binary crossentropy loss and Adam optimizer. EarlyStopping callback is employed to prevent overfitting. The training process is monitored using accuracy as the metric.

Evaluation metrics include:

- **Accuracy Score:** Accuracy of the model on the test data.
- **Confusion Matrix:** Visualization of true positive, false positive, true negative, and false negative predictions.. In addition to these, it also helps in identifying Type I errors (false positives) and Type II errors (false negatives).
- **Classification Report:** Precision, recall, F1-score, and support for each class (positive/negative).

## Model Performance Visualizations

### Confusion Matrix
![Confusion Matrix](/images/conf_mat.png)

### Word Clouds
#### Positive Reviews
![Word Cloud for Positive Reviews](/images/pos_rev.png)   

#### Negative Reviews
![Word Cloud for Negative Reviews](/images/neg_rev.png)
