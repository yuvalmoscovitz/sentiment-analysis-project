# 📈 Stock Sentiment Analysis Dashboard

## 🌐 Overview

This project aims to provide real-time sentiment analysis on publicly traded companies. The focus is on the quality of the data and the accuracy of the machine learning model used for sentiment analysis.

## 📊 Data

### 📡 Real-Time News Data

Real-time news data is sourced from Alpha Vantage and processed for sentiment analysis.

### 📚 Training Data

The machine learning model is trained on a dataset of financial news articles. The dataset undergoes rigorous cleaning and preprocessing to improve the model's performance.

#### 🧹 Data Cleaning and Preprocessing

- **🔠 Tokenization**: Sentences are tokenized into individual words.
- **🧼 Text Cleaning**: All text is converted to lowercase, and non-alphanumeric characters are removed.
- **✂️ Stemming**: Words are stemmed using the Porter Stemming algorithm.

## 🤖 Model

### 🛠 Algorithm and Libraries

The model uses a Random Forest classifier implemented with scikit-learn and leverages Word2Vec for feature extraction, implemented using the Gensim library.

### 🎛 Feature Extraction

Word2Vec is used to convert sentences into vectors, serving as features for the model.

### 🏋️‍♀️ Model Training

The model is trained using 80% of the data, with the remaining 20% reserved for testing. The model is then saved for future use.

## 🖥 Interface

The project includes a minimalistic interface for users to input a stock ticker and receive sentiment analysis results.

## 🚀 Usage

To run the project:

1. 📦 Clone the repository
2. 🛠 Install dependencies
3. 🏃‍♂️ Run `flask run`

## 👨‍💻 Developed By

Developed by Yuval Moscovitz
