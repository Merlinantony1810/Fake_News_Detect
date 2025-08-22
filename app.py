import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Paste a news **title/text** and get a prediction (Fake vs. Real).")

# Load artifacts
@st.cache_resource
def load_baseline():
    try:
        tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
        lr = joblib.load("artifacts/lr_model.pkl")
        return tfidf, lr
    except Exception as e:
        return None, None

@st.cache_resource
def load_lstm():
    try:
        with open("artifacts/tokenizer.json") as f:
            tok_json = f.read()
        tokenizer = Tokenizer.from_json(tok_json)
        model = load_model("artifacts/lstm_model.h5")
        return tokenizer, model
    except Exception as e:
        return None, None

tfidf, lr = load_baseline()
tokenizer, lstm_model = load_lstm()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text

user_title = st.text_input("Title (optional)")
user_text = st.text_area("Article Text", height=200, placeholder="Paste article text here...")

use_lstm = st.toggle("Use LSTM model (if available)", value=False)
predict_btn = st.button("Predict")

if predict_btn:
    full_text = (user_title + " " + user_text).strip()
    if len(full_text) < 10:
        st.warning("Please enter at least 10 characters of text.")
    else:
        clean = clean_text(full_text)

        if use_lstm and tokenizer and lstm_model:
            MAX_LEN = 400
            seq = tokenizer.texts_to_sequences([clean])
            pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
            proba = float(lstm_model.predict(pad, verbose=0).ravel()[0])
            pred = int(proba >= 0.5)
            st.subheader("Prediction (LSTM)")
            st.write(f"**{'Real' if pred==1 else 'Fake'}**  —  confidence: {proba if pred==1 else 1-proba:.2f}")
        elif tfidf and lr:
            X = tfidf.transform([clean])
            proba = float(lr.predict_proba(X)[0,1])
            pred = int(proba >= 0.5)
            st.subheader("Prediction (Baseline)")
            st.write(f"**{'Real' if pred==1 else 'Fake'}**  —  confidence: {proba if pred==1 else 1-proba:.2f}")

            # Simple explanation: show top weighted tokens present in text
            try:
                feature_names = np.array(tfidf.get_feature_names_out())
                coefs = lr.coef_.ravel()
                # Get indices of tokens present in this doc
                X_coo = X.tocoo()
                present_idx = np.unique(X_coo.col)
                scores = [(feature_names[i], coefs[i]) for i in present_idx]
                # Sort by importance towards predicted class
                if pred == 1:
                    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
                else:
                    scores = sorted(scores, key=lambda x: x[1])[:10]
                st.markdown("**Top contributing tokens:**")
                for w, s in scores:
                    st.write(f"- `{w}`  (weight: {s:.3f})")
            except Exception as e:
                pass
        else:
            st.error("Models not found. Please train the models and ensure artifacts exist in the `artifacts/` directory.")