"""
AI Customer Support Copilot — ZENDS Communications
Streamlit application implementing the full pipeline from the project brief:
  Input -> Preprocessing -> Intent Classification -> Sentiment Analysis ->
  Knowledge Base Retrieval (RAG) -> Response Generation -> Agent Review

Run with:
    streamlit run app.py
"""

import os
import pickle
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import importlib

rag_engine = importlib.import_module("05_rag_engine")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="ZENDS AI Support Copilot", page_icon="📡", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

with st.sidebar:
    st.info(
        "**Environment note:** this deployment uses lightweight, fully-tested "
        "substitutes for the brief's specified HuggingFace transformer models "
        "and open-source LLM (TF-IDF+LogisticRegression for intent, VADER for "
        "sentiment, TF-IDF+FAISS for retrieval, and template-based generation) "
        "since this environment cannot download models from huggingface.co. "
        "See README.md 'Environment Limitations' for the brief-spec scripts "
        "meant to run where model downloads are available.",
        icon="ℹ️",
    )


@st.cache_resource
def load_models():
    with open(f"{MODELS_DIR}/intent_vectorizer.pkl", "rb") as f:
        intent_vectorizer = pickle.load(f)
    with open(f"{MODELS_DIR}/intent_classifier.pkl", "rb") as f:
        intent_clf = pickle.load(f)

    index, kb_vectorizer, chunks = rag_engine.load_rag_artifacts(
        f"{MODELS_DIR}/knowledge_base.index",
        f"{MODELS_DIR}/kb_vectorizer.pkl",
        f"{MODELS_DIR}/kb_chunks.pkl",
    )

    sentiment_analyzer = SentimentIntensityAnalyzer()

    return intent_vectorizer, intent_clf, index, kb_vectorizer, chunks, sentiment_analyzer


intent_vectorizer, intent_clf, kb_index, kb_vectorizer, kb_chunks, sentiment_analyzer = load_models()


def predict_intent(text: str) -> str:
    vec = intent_vectorizer.transform([text])
    return intent_clf.predict(vec)[0]


def predict_sentiment(text: str) -> str:
    score = sentiment_analyzer.polarity_scores(text)["compound"]
    if score <= -0.3:
        return "angry"
    elif score >= 0.3:
        return "happy"
    return "neutral"


st.title("📡 ZENDS AI Customer Support Copilot")
st.caption("Intent detection → Sentiment analysis → Policy retrieval (RAG) → Suggested reply for agent review")

col_input, col_output = st.columns([1, 1.3])

with col_input:
    st.subheader("Customer Message")
    example_queries = [
        "Why is my bill 200 for Postpaid Gold?",
        "Can you process a refund for ZENDCloud VM Pro?",
        "I can't access my Prepaid Plus.",
        "Does Prepaid Basic support 5G data?",
        "This is the third time I've had this issue, completely unacceptable!",
    ]
    chosen_example = st.selectbox("Try an example, or type your own below:", ["(type your own)"] + example_queries)

    default_text = "" if chosen_example == "(type your own)" else chosen_example
    customer_message = st.text_area("Message", value=default_text, height=120)

    top_k = st.slider("Number of policy/product sources to retrieve", 1, 5, 2)
    analyze = st.button("Analyze & Generate Response", type="primary")

with col_output:
    if analyze and customer_message.strip():
        with st.spinner("Processing..."):
            intent = predict_intent(customer_message)
            sentiment = predict_sentiment(customer_message)
            context = rag_engine.get_relevant_context(
                customer_message, kb_index, kb_vectorizer, kb_chunks, top_k=top_k, intent=intent
            )
            response = rag_engine.generate_response(customer_message, intent, sentiment, context)

        st.subheader("Analysis")
        c1, c2 = st.columns(2)
        c1.metric("Predicted Intent", intent.title())
        sentiment_emoji = {"angry": "😠", "happy": "😊", "neutral": "😐"}
        c2.metric("Predicted Sentiment", f"{sentiment_emoji.get(sentiment, '')} {sentiment.title()}")

        st.subheader("Retrieved Knowledge Base Context")
        for c in context:
            with st.expander(f"📄 {c['title']}  (relevance: {c['boosted_score']:.2f})"):
                st.write(c["text"])

        st.subheader("🤖 Suggested Response (for agent review)")
        st.success(response)
        st.caption(
            "This is an AI-suggested response grounded in the retrieved policy/product context above. "
            "Please review before sending to the customer."
        )
    elif analyze:
        st.warning("Type or select a customer message first.")
    else:
        st.info("Enter a customer message and click **Analyze & Generate Response** to see the full pipeline in action.")
