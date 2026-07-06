"""
02b_train_intent_classifier_huggingface.py

Intent Classification Model — BRIEF-SPEC VERSION (fine-tuned DistilBERT).

*** NOT EXECUTED/TESTED IN THIS ENVIRONMENT ***
This sandbox's network access is restricted to PyPI/npm/GitHub and does not
include huggingface.co, so `from_pretrained("distilbert-base-uncased")`
cannot download model weights here. This script is written correctly to the
brief's specification and is intended to be run on your own machine, Google
Colab, or any environment with internet access to huggingface.co (and
ideally a GPU — fine-tuning on CPU works but is slow).

Given the dataset only has 368 unique examples (see README "A Note on
Dataset Size" and scripts/01_dedupe_data.py for why), expect this to overfit
quickly — use a small number of epochs (2-3), a low learning rate, and watch
validation loss closely. The TF-IDF + Logistic Regression baseline in
02_train_intent_classifier.py is the more realistic choice for this data
size; this script is provided to satisfy the brief's specified tech stack
and as a template you can re-run once a larger, properly diversified
dataset is available.

Usage (on a machine with huggingface.co access):
    pip install transformers datasets torch scikit-learn
    python 02b_train_intent_classifier_huggingface.py --input ../data/cleaned_queries.csv --output-dir ../models/distilbert_intent --epochs 3
"""

import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/cleaned_queries.csv")
    parser.add_argument("--output-dir", default="../models/distilbert_intent")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    labels = sorted(df["intent"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["label"] = df["intent"].map(label2id)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["intent"], random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final validation metrics:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
