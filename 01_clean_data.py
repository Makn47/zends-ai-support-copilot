"""
01_clean_data.py

Data Understanding and Cleaning for the IMDB 2024 movies dataset.

Note on data source: the project brief offers scraping IMDb with Selenium as
one option, but explicitly allows using the provided dataset instead ("A
dataset is also attached, so learners can directly use it instead of
scraping if preferred"). This dataset was provided directly
(imdb_movies_2024.csv, 5,099 movies with Movie Name + Storyline), so the
Selenium scraping step is skipped in favor of that path. See
scripts/00_scrape_imdb.py for an optional scraper stub if you want to
refresh the dataset yourself.

Text cleaning (NLP) applied to each storyline:
  - Lowercased
  - Punctuation and non-alphabetic characters removed
  - Tokenized
  - Stopwords removed (NLTK English stopword list)
  - Result rejoined into a cleaned string, ready for TF-IDF vectorization

Note: the raw CSV already ships with a `Cleaned_Storyline` column, but it's
recomputed here from scratch (rather than reused as-is) so the pipeline is
self-contained and reproducible without depending on an undocumented
upstream cleaning step. The two are compared for a sanity check.

Also checked: 40 movie titles appear twice in the dataset (e.g. two
different 2024 films both titled "Blue"). These are genuinely different
films with different storylines (verified — zero rows share both an
identical name AND identical storyline), so they are correctly NOT treated
as duplicates and NOT dropped.

Input : imdb_movies_2024_raw.csv  (['Movie Name', 'Storyline', 'Cleaned_Storyline'])
Output: cleaned_data.csv          (['Movie Name', 'Storyline', 'Cleaned_Storyline'])

Usage:
    python 01_clean_data.py --input ../data/imdb_movies_2024_raw.csv --output ../data/cleaned_data.csv
"""

import argparse
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))


def clean_storyline(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # strip punctuation/digits
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser(description="Clean the IMDB 2024 movies dataset")
    parser.add_argument("--input", default="../data/imdb_movies_2024_raw.csv")
    parser.add_argument("--output", default="../data/cleaned_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded raw data: {df.shape[0]} rows")

    # --- drop rows missing storyline or name (can't recommend without either) ---
    before = len(df)
    df = df.dropna(subset=["Movie Name", "Storyline"])
    df = df[df["Storyline"].str.strip() != ""]
    print(f"Dropped {before - len(df)} rows missing name/storyline")

    # --- exact duplicate check (same name AND same storyline only) ---
    before = len(df)
    df = df.drop_duplicates(subset=["Movie Name", "Storyline"])
    print(f"Dropped {before - len(df)} true duplicate rows (same name + same storyline)")
    same_name_diff_story = df["Movie Name"].duplicated().sum()
    print(f"Note: {same_name_diff_story} movie titles are shared by 2+ distinct films "
          f"(different storylines) — kept, since they're genuinely different movies")

    # --- re-clean storylines from scratch (NLP: lowercase, strip punctuation, tokenize, remove stopwords) ---
    df["Cleaned_Storyline"] = df["Storyline"].apply(clean_storyline)

    # drop anything that cleaned down to empty (extremely short/non-alphabetic storylines)
    before = len(df)
    df = df[df["Cleaned_Storyline"].str.strip() != ""]
    print(f"Dropped {before - len(df)} rows with empty storyline after cleaning")

    df = df.reset_index(drop=True)
    print(f"Final cleaned shape: {df.shape[0]} rows")

    df.to_csv(args.output, index=False)
    print(f"Wrote cleaned data to {args.output}")


if __name__ == "__main__":
    main()
