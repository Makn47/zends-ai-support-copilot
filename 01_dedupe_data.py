"""
01_dedupe_data.py

Data Cleaning: deduplicates the synthetic customer query dataset.

Root cause of the duplication (documented for transparency): the original
generation notebook (zends_Communications_Synthetic_Generation_.ipynb) selects
templates and entities using `i % len(list)` with short lists (3 templates,
8 products, 4 countries, 5 amounts). This cycles through the same small set
of combinations long before reaching the target row count, and the
paraphrasing step (a T5 paraphrase model) did not add enough lexical
variety on short template-like sentences to compensate. The result: out of
10,000 rows, only 368 are actually unique text values.

Rather than fine-tune models on a dataset that is 96.3% exact duplicates
(which would let a model "memorize" instead of generalize, and would leak
identical examples across any train/test split), this script deduplicates
down to the honest 368 unique examples. This is a small dataset for 5-class
intent + 3-class sentiment classification — that limitation is real and is
documented in the README rather than hidden. See README.md, section
"A Note on Dataset Size", for how this shaped the modeling approach.

Usage:
    python 01_dedupe_data.py --input ../data/zends_queries_raw.csv --output ../data/cleaned_queries.csv
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Deduplicate the ZENDS synthetic customer query dataset")
    parser.add_argument("--input", default="../data/zends_queries_raw.csv")
    parser.add_argument("--output", default="../data/cleaned_queries.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded raw data: {len(df)} rows")

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows ({(before - len(df)) / before:.1%} of the dataset)")
    print(f"Remaining unique examples: {len(df)}")

    print("\nIntent distribution:")
    print(df["intent"].value_counts())
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    df.to_csv(args.output, index=False)
    print(f"\nWrote deduplicated data to {args.output}")


if __name__ == "__main__":
    main()
