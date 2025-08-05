# 🎬 IMDB Sentiment Analysis using BiLSTM (with TensorFlow Embedding)

This project implements a sentiment analysis model on the IMDB movie reviews dataset using TensorFlow and Keras. It uses a Bidirectional LSTM with an embedding layer to classify reviews as **positive** or **negative**.

---

## 📚 Dataset: IMDB Movie Reviews

- The [IMDB Movie Reviews Dataset]((https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)) is a widely used benchmark for sentiment classification.
- It contains **50,000 movie reviews**, split equally into **25,000 Positive** and **25,000 Negative** samples.
- Labels: `0` → **Negative**, `1` → **Positive**
- Reviews are **preprocessed and encoded as integers**, where each integer represents a word from a dictionary.

# Project Features


Load and preprocess IMDB data

Use TensorFlow's built-in Embedding layer

Build a Bidirectional LSTM model for sequence learning

Apply dropout regularization

Evaluate on test data and plot training history

---

# Web Application (Built with Streamlit)
We have built an interactive web application using Streamlit that allows users to:

Submit their own reviews

View sentiment prediction (Positive / Negative)

See the prediction confidence score


To Run Locally:


pip install streamlit

streamlit run app.py


---

# Model Architecture


## Input → Embedding Layer → BiLSTM (128 units) → Dropout → Dense → Sigmoid

Embedding: Converts integer word indices to dense vectors

Bidirectional LSTM: Captures context from both directions

Dropout: Prevents overfitting

Dense: Outputs probability for binary classification


# Technologies Used

Python 

TensorFlow / Keras

NumPy, Pandas, Matplotlib

Colab, VSCode

---

# How to Run the Project


1. Clone the Repository

git clone https://github.com/Pratyush-Basu/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis

2. Install Required Libraries

pip install -r requirements.txt

Or install manually:


pip install tensorflow numpy matplotlib pandas


# Run the Notebook
Use Jupyter or Colab to run the IMDB_With_bi_Directional_LSTM_NEW.ipynb file.


---


# Sample Output



Epoch 1/5
loss: 0.5039 - accuracy: 0.7410

Epoch 5/5
loss: 0.1524 - accuracy: 0.9478

Test Accuracy: 0.92


---


# Visualizations

Training & Validation Accuracy

Training & Validation Loss

Included at the end of the notebook using Matplotlib

---


# Improvements


Add custom tokenizer for raw text input

Use pre-trained embeddings (e.g., GloVe)

Add attention mechanism

Hyperparameter tuning (batch size, LSTM units)

# Example Use Case
Use this model to:

Analyze customer feedback

Classify product reviews

Understand movie or app review sentiments


# Author
Pratyush Basu



