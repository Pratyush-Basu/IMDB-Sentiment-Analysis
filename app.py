import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = tf.keras.models.load_model('imdb_sentiment_analysis_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


MAX_LEN = 1431

# Streamlit app UI
st.title('🎬 IMDB Sentiment Analysis')
st.write("This app predicts whether a movie review is **positive** or **negative**.")

# Input text box
review = st.text_area("✏️ Enter your movie review here:")

if st.button('Predict Sentiment'):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review to predict.")
    else:
        
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded)[0][0]  # probability

        # Display result
        if prediction >= 0.5:
            st.success(f"✅ **Positive review**  (Confidence: {prediction:.2f})")
        else:
            st.error(f"⚠️ **Negative review**  (Confidence: {1-prediction:.2f})")

st.markdown("""
---
Built with ❤️ by Pratyush Basu.
""")
#.\venv\Scripts\activate
#Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
#python -m streamlit run app.py