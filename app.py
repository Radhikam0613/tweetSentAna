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


