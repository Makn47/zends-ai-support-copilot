"""
02_train_intent_classifier.py

Intent Classification Model — LITE VERSION (fully tested in this environment).

Why TF-IDF + Logistic Regression instead of fine-tuning DistilBERT/BERT/RoBERTa
as specified in the brief: with only 368 unique examples across 5 classes
(and as few as 48 examples for some classes), fine-tuning a full transformer
would almost certainly overfit badly — transformer fine-tuning typically
needs thousands of examples per class to generalize well. A linear model
over TF-IDF features is the more honest and appropriate choice for this data
size, and is standard practice for small-data text classification.

For the actual brief-specified approach, see
02b_train_intent_classifier_huggingface.py — a complete DistilBERT
fine-tuning script, written to spec but UNTESTED in this environment (no
route to huggingface.co here; see README "Environment Limitations").

Evaluation approach: with so few examples per class, a single train/test
split gives a noisy, unreliable estimate (e.g. the 32-example 'happy'
sentiment class would leave only ~6 examples in a 20% test split). Stratified
K-Fold cross-validation is used instead for the headline accuracy/F1 metrics,
with one held-out split kept only to render a representative confusion matrix
for the deliverable.

Usage:
    python 02_train_intent_classifier.py --input ../data/cleaned_queries.csv --output-dir ../models
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/cleaned_queries.csv")
    parser.add_argument("--output-dir", default="../models")
    parser.add_argument("--target", default="intent", choices=["intent", "sentiment"])
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X_text = df["text"]
    y = df[args.target]

    print(f"Training {args.target} classifier on {len(df)} examples")
    print(y.value_counts())

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(X_text)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    # --- Stratified K-Fold cross-validation (more reliable than a single split at this sample size) ---
    min_class_count = y.value_counts().min()
    n_splits = min(5, min_class_count)  # can't have more folds than the smallest class
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_preds = cross_val_predict(clf, X, y, cv=skf)
    cv_f1 = f1_score(y, cv_preds, average="macro")
    print(f"\n{n_splits}-fold CV macro F1: {cv_f1:.3f}")
    print(f"\n{n_splits}-fold CV classification report:")
    print(classification_report(y, cv_preds, zero_division=0))

    # --- Fit final model on all data (for deployment) ---
    clf.fit(X, y)

    # --- One held-out split purely to render a confusion matrix for the deliverable ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    holdout_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    holdout_clf.fit(X_train, y_train)
    y_pred = holdout_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    print(f"\nHeld-out confusion matrix (labels={sorted(y.unique())}):")
    print(cm)

    with open(f"{args.output_dir}/{args.target}_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{args.output_dir}/{args.target}_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    np.save(f"{args.output_dir}/{args.target}_confusion_matrix.npy", cm)
    with open(f"{args.output_dir}/{args.target}_labels.pkl", "wb") as f:
        pickle.dump(sorted(y.unique()), f)

    print(f"\nSaved vectorizer + classifier to {args.output_dir}/{args.target}_*.pkl")


if __name__ == "__main__":
    main()
