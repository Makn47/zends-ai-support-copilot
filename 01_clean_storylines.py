"""
01_clean_storylines.py

Data Preprocessing (NLP): cleans and tokenizes movie storylines.

Steps (as specified in the project brief):
  - Lowercase
  - Remove punctuation / non-alphabetic characters
  - Remove stopwords (NLTK English stopword list)
  - Tokenize

Note: the supplied dataset (imdb_movies_2024.csv) already ships with a
pre-cleaned 'Cleaned_Storyline' column. This script independently reproduces
that cleaning step from the raw 'Storyline' column (rather than just reusing
the provided one) to satisfy the brief's explicit "Data Preprocessing (NLP)"
deliverable, and cross-checks its output against the shipped column as a
validation step.

Input : imdb_movies_2024_raw.csv  (columns: 'Movie Name', 'Storyline', 'Cleaned_Storyline')
Output: cleaned_movies.csv        (columns: 'movie_name', 'storyline', 'cleaned_storyline')

Usage:
    python 01_clean_storylines.py --input ../data/imdb_movies_2024_raw.csv --output ../data/cleaned_movies.csv
"""

import argparse
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # strip punctuation/digits
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser(description="Clean and tokenize movie storylines")
    parser.add_argument("--input", default="../data/imdb_movies_2024_raw.csv")
    parser.add_argument("--output", default="../data/cleaned_movies.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} movies")

    # --- drop rows missing essential fields ---
    before = len(df)
    df = df.dropna(subset=["Movie Name", "Storyline"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows missing movie name / storyline")

    # --- duplicate check: same title CAN be a different film (verified: 40 shared
    # titles in this dataset all have distinct storylines), so only exact full-row
    # duplicates are dropped, not same-title rows. ---
    before = len(df)
    df = df.drop_duplicates(subset=["Movie Name", "Storyline"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} exact duplicate (name + storyline) rows")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["cleaned_storyline_generated"] = df["storyline"].apply(clean_text)

    # sanity check against the dataset's own pre-cleaned column, if present
    if "cleaned_storyline" in df.columns:
        sample_match = (
            df["cleaned_storyline_generated"].str.split().str[:5]
            == df["cleaned_storyline"].str.split().str[:5]
        ).mean()
        print(f"First-5-token agreement with provided Cleaned_Storyline: {sample_match:.1%}")

    out = df[["movie_name", "storyline", "cleaned_storyline_generated"]].rename(
        columns={"cleaned_storyline_generated": "cleaned_storyline"}
    )

    # drop rows where cleaning left an empty string (all-stopword or non-alphabetic storyline)
    before = len(out)
    out = out[out["cleaned_storyline"].str.strip() != ""].reset_index(drop=True)
    print(f"Dropped {before - len(out)} rows with empty cleaned storyline")

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} cleaned movies to {args.output}")


if __name__ == "__main__":
    main()
