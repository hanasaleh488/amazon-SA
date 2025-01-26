import streamlit as st
import pickle
import os

# Load the model and vectorizer
model = pickle.load(open(os.path.join('model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join('tfidf_vectorizer.pkl'), 'rb'))

st.title("Sentiment Analysis")

review_input = st.text_area("Enter your review:")
if st.button("Predict"):
    if review_input.strip():
        review_tfidf = vectorizer.transform([review_input])
        prediction = model.predict(review_tfidf)[0]
        st.write("Positive" if prediction == 1 else "Negative")
