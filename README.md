# Fake_News_Detection

## Overview
This project implements a **Fake News Detection system** using both classical and deep learning methods. The goal is to classify news articles as **real** or **fake** based on their textual content.  

We use the Kaggle [Fake & Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), combining **EDA, preprocessing, baseline modeling (TF–IDF + Logistic Regression), and deep learning (LSTM)** approaches.

## Features
- **Data Cleaning:** Lowercasing, stopword removal, lemmatization, combining title & body text.  
- **Exploratory Data Analysis:** Word frequency visualization, word clouds, and text length statistics.  
- **Baseline Model:** TF–IDF vectorization + Logistic Regression — interpretable and fast.  
- **Deep Model:** LSTM Neural Network — captures sequential context in text.  
- **Evaluation:** Accuracy, ROC-AUC, confusion matrix, classification report.

## Dataset
- `True.csv` — Real news articles  
- `Fake.csv` — Fake news articles  
- Each dataset includes `title`, `text`, `subject`, and `date` columns.  
- Combined and labeled: `1` for real, `0` for fake.

## Results
- Logistic Regression achieved ~99% accuracy and high ROC-AUC.
- LSTM achieved similar performance and can generalize better with more data.
- Both models are ready for deployment.

