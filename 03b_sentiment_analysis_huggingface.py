"""
03b_sentiment_analysis_huggingface.py

Sentiment Analysis Model — BRIEF-SPEC VERSION (pretrained HuggingFace pipeline).

*** NOT EXECUTED/TESTED IN THIS ENVIRONMENT *** (no huggingface.co access
in this sandbox — see README "Environment Limitations"). Written to spec
for you to run wherever huggingface.co is reachable.

Uses a pretrained sentiment pipeline (no fine-tuning needed) exactly as
listed in the brief. Note the label mapping: these models typically output
POSITIVE/NEGATIVE (2-class) or Positive/Negative/Neutral depending on the
checkpoint — the mapping below assumes distilbert-base-uncased-finetuned-
sst-2-english (2-class: POSITIVE/NEGATIVE) and maps to this project's
angry/happy/neutral scheme using a confidence threshold for neutral.
Swap in cardiffnlp/twitter-roberta-base-sentiment for native 3-class output
if you prefer not to threshold.

Usage (on a machine with huggingface.co access):
    pip install transformers torch
    python 03b_sentiment_analysis_huggingface.py --input ../data/cleaned_queries.csv
"""

import argparse

import pandas as pd
from transformers import pipeline


def map_to_project_labels(hf_result: dict, neutral_confidence_threshold: float = 0.65) -> str:
    """Map a 2-class POSITIVE/NEGATIVE HF result to this project's angry/happy/neutral scheme."""
    label, score = hf_result["label"], hf_result["score"]
    if score < neutral_confidence_threshold:
        return "neutral"
    return "happy" if label == "POSITIVE" else "angry"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/cleaned_queries.csv")
    parser.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    sentiment_pipeline = pipeline("sentiment-analysis", model=args.model)
    raw_results = sentiment_pipeline(df["text"].tolist(), truncation=True)
    df["hf_sentiment"] = [map_to_project_labels(r) for r in raw_results]

    print(df[["text", "sentiment", "hf_sentiment"]].head(20))
    agreement = (df["sentiment"] == df["hf_sentiment"]).mean()
    print(f"\nAgreement with dataset's VADER-derived labels: {agreement:.1%}")
    print("(Some disagreement is expected and healthy here — VADER and a transformer")
    print(" sentiment model use different criteria, unlike the circular VADER-vs-VADER")
    print(" comparison in 03_sentiment_analysis.py.)")


if __name__ == "__main__":
    main()
