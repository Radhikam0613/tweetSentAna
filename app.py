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

st.set_page_config(layout="wide")

# ================================
# Load model
# ================================
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ================================
# PDF extraction
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
st.title("📊 PDF Sentiment Dashboard")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Analyze PDF"):

        raw_text = extract_text_from_pdf(uploaded_file)
        reactions = [line.strip() for line in raw_text.split("\n") if line.strip() != ""]

        if len(reactions) == 0:
            st.error("No text found.")
        else:
            vectors = vectorizer.transform(reactions)
            predictions = model.predict(vectors)
            probabilities = model.predict_proba(vectors)

            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiments = [sentiment_map.get(p, 'Unknown') for p in predictions]
            confidence = np.max(probabilities, axis=1)

            df = pd.DataFrame({
                "Reaction": reactions,
                "Sentiment": sentiments,
                "Confidence": confidence
            })

            counts = Counter(sentiments)
            total = len(sentiments)

            pos = counts.get("Positive", 0)
            neu = counts.get("Neutral", 0)
            neg = counts.get("Negative", 0)

            # ================================
            # TOP: SUMMARY + VERDICT
            # ================================
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total", total)
            col2.metric("Positive", f"{pos} ({round(pos/total*100,2)}%)")
            col3.metric("Neutral", f"{neu} ({round(neu/total*100,2)}%)")
            col4.metric("Negative", f"{neg} ({round(neg/total*100,2)}%)")

            # Overall sentiment
            if pos > neg and pos > neu:
                overall = "Positive 😊"
            elif neg > pos and neg > neu:
                overall = "Negative 😡"
            else:
                overall = "Neutral 😐"

            st.markdown(f"### 🧠 Overall Sentiment: **{overall}**")

            # ================================
            # MIDDLE: CHARTS SIDE BY SIDE
            # ================================
            c1, c2 = st.columns(2)

            labels = ["Positive", "Neutral", "Negative"]
            values = [pos, neu, neg]

            # Bar Chart
            fig1, ax1 = plt.subplots()
            ax1.bar(labels, values)
            ax1.set_title("Distribution")

            c1.pyplot(fig1)

            # Pie Chart
            fig2, ax2 = plt.subplots()
            ax2.pie(values, labels=labels, autopct='%1.1f%%')
            ax2.set_title("Share")

            c2.pyplot(fig2)

            # ================================
            # BOTTOM: TABLE
            # ================================
            st.markdown("### 📄 Detailed Predictions")
            st.dataframe(df, use_container_width=True)
