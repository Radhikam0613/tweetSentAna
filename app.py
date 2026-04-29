# import streamlit as st
# import pickle

# # Load the saved vectorizer and model
# with open('tfidf_vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)

# with open('logreg_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Streamlit app title
# st.title("Tweet Sentiment Analyzer")
# st.write("Enter a tweet and find out if it's positive, negative, or neutral!")

# # Input box for tweet
# tweet = st.text_input("Enter your tweet here:")

# if st.button("Predict Sentiment"):
#     if tweet.strip() == "":
#         st.warning("Please enter a tweet!")
#     else:
#         # Vectorize and predict
#         tweet_vec = vectorizer.transform([tweet])
#         pred_label = model.predict(tweet_vec)[0]

#         # Map numeric label to sentiment
#         sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
#         sentiment = sentiment_map.get(pred_label, 'Unknown')

#         st.success(f"Predicted Sentiment: {sentiment}")

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import PyPDF2
import matplotlib.pyplot as plt

# ================================
# Load model + vectorizer
# ================================
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ================================
# PDF Text Extraction
# ================================
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


# ================================
# UI
# ================================
st.title("PDF Sentiment Analyzer (Now with Graphs™)")
st.write("Upload your PDF. I’ll turn opinions into statistics.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Analyze PDF"):

        raw_text = extract_text_from_pdf(uploaded_file)
        reactions = [line.strip() for line in raw_text.split("\n") if line.strip() != ""]

        if len(reactions) == 0:
            st.error("No text found. That PDF was emotionally empty.")
        else:
            # Vectorize
            vectors = vectorizer.transform(reactions)

            # Predictions
            predictions = model.predict(vectors)
            probabilities = model.predict_proba(vectors)

            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiments = [sentiment_map.get(p, 'Unknown') for p in predictions]

            # Confidence = max probability
            confidence = np.max(probabilities, axis=1)

            # ================================
            # DataFrame
            # ================================
            df = pd.DataFrame({
                "Reaction": reactions,
                "Sentiment": sentiments,
                "Confidence": confidence
            })

            st.subheader("📄 Predictions with Confidence")
            st.dataframe(df)

            # ================================
            # Metrics
            # ================================
            counts = Counter(sentiments)
            total = len(sentiments)

            metrics = {
                "Total": total,
                "Positive %": round((counts.get("Positive", 0) / total) * 100, 2),
                "Neutral %": round((counts.get("Neutral", 0) / total) * 100, 2),
                "Negative %": round((counts.get("Negative", 0) / total) * 100, 2),
                "Avg Confidence": round(float(np.mean(confidence)), 3)
            }

            st.subheader("📊 Summary")
            st.json(metrics)

            # ================================
            # BAR CHART
            # ================================
            st.subheader("📊 Bar Chart")

            labels = list(counts.keys())
            values = list(counts.values())

            fig1, ax1 = plt.subplots()
            ax1.bar(labels, values)
            ax1.set_title("Sentiment Distribution")
            ax1.set_xlabel("Sentiment")
            ax1.set_ylabel("Count")

            st.pyplot(fig1)

            # ================================
            # PIE CHART
            # ================================
            st.subheader("🥧 Pie Chart")

            fig2, ax2 = plt.subplots()
            ax2.pie(values, labels=labels, autopct='%1.1f%%')
            ax2.set_title("Sentiment Share")

            st.pyplot(fig2)
