# import streamlit as st
# import pandas as pd
# import pickle
# import re

# # Load model and vectorizer
# @st.cache_resource
# def load_all():
#     with open("model.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("vectorizer.pkl", "rb") as f:
#         vectorizer = pickle.load(f)
#     return model, vectorizer

# model, vectorizer = load_all()

# label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r'@\w+|\#', '', text)
#     text = re.sub(r'[^a-z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# st.title("Twitter Sentiment Analysis")

# user_input = st.text_area("Enter a tweet:")

# if st.button("Predict Sentiment"):
#     clean = clean_text(user_input)
#     vec = vectorizer.transform([clean])
#     pred = model.predict(vec)[0]
#     st.write(f"**Predicted Sentiment:** {label_map.get(pred, 'Unknown')}")



import streamlit as st
import pickle

# Load the saved vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app title
st.title("Tweet Sentiment Analyzer")
st.write("Enter a tweet and find out if it's positive, negative, or neutral!")

# Input box for tweet
tweet = st.text_input("Enter your tweet here:")

if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet!")
    else:
        # Vectorize and predict
        tweet_vec = vectorizer.transform([tweet])
        pred_label = model.predict(tweet_vec)[0]

        # Map numeric label to sentiment
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_map.get(pred_label, 'Unknown')

        st.success(f"Predicted Sentiment: {sentiment}")

