"""
recommend.py

Storyline-based movie recommendation engine.

Two entry points, both computing cosine similarity on demand against the
precomputed TF-IDF matrix (no need to precompute the full NxN matrix — see
02_build_similarity_model.py for why):

  1. recommend_from_storyline(user_text, ...)
     The primary use case from the brief: a user types a short storyline
     description, and the system returns the top-N most similar existing
     movies. The input text is cleaned with the same text_clean() function
     used during training, transformed with the fitted TF-IDF vectorizer,
     and compared against every movie's TF-IDF vector.

  2. recommend_similar_to_movie(movie_name, ...)
     Complementary "more like this" feature: given an existing movie in the
     dataset, find other movies with the most similar storyline.
"""

import pickle
import re

import pandas as pd
import scipy.sparse as sp
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Same cleaning logic as scripts/01_clean_storylines.py, kept in sync for consistent vectorization."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def load_artifacts(movies_path, vectorizer_path, tfidf_matrix_path):
    movies = pd.read_csv(movies_path)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = sp.load_npz(tfidf_matrix_path)
    return movies, vectorizer, tfidf_matrix


def recommend_from_storyline(
    user_text: str, movies: pd.DataFrame, vectorizer, tfidf_matrix, top_n: int = 5
) -> pd.DataFrame:
    """Given a user-typed storyline, return the top_n most similar movies."""
    cleaned = clean_text(user_text)
    if not cleaned:
        return movies.iloc[0:0]

    user_vec = vectorizer.transform([cleaned])
    similarities = cosine_similarity(user_vec, tfidf_matrix)[0]

    result = movies.copy()
    result["similarity"] = similarities
    result = result.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
    return result


def recommend_similar_to_movie(
    movie_name: str, movies: pd.DataFrame, tfidf_matrix, top_n: int = 5
) -> pd.DataFrame:
    """Given an existing movie's name, return the top_n most similar OTHER movies."""
    matches = movies.index[movies["movie_name"] == movie_name].tolist()
    if not matches:
        return movies.iloc[0:0]
    idx = matches[0]

    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]

    result = movies.copy()
    result["similarity"] = similarities
    result = result.drop(index=idx)  # exclude the movie itself
    result = result.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
    return result


if __name__ == "__main__":
    # quick smoke test
    movies, vectorizer, tfidf_matrix = load_artifacts(
        "../data/cleaned_movies.csv", "../models/tfidf_vectorizer.pkl", "../models/tfidf_matrix.npz"
    )

    print("=== Test 1: recommend from a typed storyline ===")
    query = "A young wizard begins his journey at a magical school where he makes friends and enemies, facing dark forces along the way."
    recs = recommend_from_storyline(query, movies, vectorizer, tfidf_matrix, top_n=5)
    print(recs[["movie_name", "similarity"]])

    print("\n=== Test 2: recommend similar to an existing movie ===")
    sample_movie = movies["movie_name"].iloc[0]
    print(f"Movies similar to: {sample_movie}")
    recs2 = recommend_similar_to_movie(sample_movie, movies, tfidf_matrix, top_n=5)
    print(recs2[["movie_name", "similarity"]])
