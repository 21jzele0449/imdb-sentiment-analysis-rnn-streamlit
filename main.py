# step 1: import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# load dataset index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# load model
model = load_model('simplernn_model.h5')

# helper function
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(i, 0) for i in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# STREAMLIT UI (OUTSIDE FUNCTION)
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict sentiment")

user_input = st.text_area("Movie Review")

if st.button('Classify'):
    processed_input = preprocess_text(user_input)
    prediction = model.predict(processed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write("Please enter a review and click the classify button")