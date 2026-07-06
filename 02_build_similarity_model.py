"""
02_build_similarity_model.py

Text Representation + Similarity: converts cleaned storylines into TF-IDF
vectors and precomputes the full movie-to-movie cosine similarity matrix.

Why NOT precompute the full movie-to-movie similarity matrix:
  A first version of this script precomputed the full 5099 x 5099 similarity
  matrix, but at ~100MB (float32) that sits right at GitHub's file-size
  ceiling for no real benefit: computing cosine similarity for ONE movie (or
  one user-typed storyline) against the 5099 x 10000 TF-IDF matrix is a
  single sparse matrix-vector product — a few milliseconds — so there's
  nothing to gain from precomputing all 26M pairwise scores when only one
  row is ever needed per request. The app computes similarity on demand
  against the saved TF-IDF matrix instead (see scripts/recommend.py), which
  satisfies the brief's "Optimized Queries" guideline more directly than a
  bulky precomputed matrix would.

Output:
  - tfidf_vectorizer.pkl : fitted TfidfVectorizer
  - tfidf_matrix.npz     : sparse TF-IDF matrix (5099 x vocab_size)

Usage:
    python 02_build_similarity_model.py --input ../data/cleaned_movies.csv --output-dir ../models
"""

import argparse
import pickle

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/cleaned_movies.csv")
    parser.add_argument("--output-dir", default="../models")
    parser.add_argument("--max-features", type=int, default=10000)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} cleaned movies")

    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_storyline"].fillna(""))
    print(f"TF-IDF matrix: {tfidf_matrix.shape[0]} movies x {tfidf_matrix.shape[1]} features")

    with open(f"{args.output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    sp.save_npz(f"{args.output_dir}/tfidf_matrix.npz", tfidf_matrix)

    print(f"Saved vectorizer and TF-IDF matrix to {args.output_dir}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
