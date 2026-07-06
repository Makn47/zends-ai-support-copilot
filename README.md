# IMDB Movie Recommendation System Using Storylines

A content-based movie recommender that suggests similar films based on storyline text, using TF-IDF vectorization and cosine similarity, with an interactive Streamlit interface.

**Domain:** Entertainment / Data Analytics / Recommender Systems
**Skills:** Python, Pandas, NLP (TF-IDF, stopword removal), Cosine Similarity, Streamlit

---

## Problem Statement

Extract 2024 movie data (name + storyline), preprocess the text with NLP techniques, represent it numerically with TF-IDF, and recommend the top 5 most similar movies to a user-provided storyline via an interactive Streamlit app.

## Project Structure

```
.
├── data/
│   ├── imdb_movies_2024_raw.csv   # supplied dataset (5,099 movies)
│   └── cleaned_movies.csv          # cleaned + tokenized storylines
├── models/
│   ├── tfidf_vectorizer.pkl        # fitted TfidfVectorizer
│   └── tfidf_matrix.npz            # sparse TF-IDF matrix (5099 x 10000)
├── scripts/
│   ├── 00_scrape_imdb_optional.py  # optional Selenium scraper (not run — see note below)
│   ├── 01_clean_storylines.py      # NLP text cleaning
│   ├── 02_build_similarity_model.py# TF-IDF vectorization
│   └── recommend.py                # recommendation engine (cosine similarity)
├── streamlit_app/
│   └── app.py                      # interactive dashboard
├── requirements.txt
└── README.md
```

## Data Source

The brief supplied `imdb_movies_2024.csv` (5,099 movies, columns: Movie Name, Storyline, Cleaned_Storyline) as an alternative to scraping IMDb directly — this project uses that dataset. `scripts/00_scrape_imdb_optional.py` is included to satisfy the brief's "Scraping Script" deliverable and to document how the dataset could be refreshed or extended from IMDb directly using Selenium, but it was not executed for this submission.

**Data validation performed:**
- No missing values in Movie Name or Storyline.
- 40 movie titles appear twice — verified these are genuinely **different films that happen to share a title** (e.g. two unrelated 2024 movies both named "A Family Affair"), not duplicate records, so they were kept rather than dropped.
- 0 exact duplicate (name + storyline) rows.

## NLP Preprocessing

`scripts/01_clean_storylines.py` independently reproduces the cleaning pipeline (rather than just reusing the dataset's supplied `Cleaned_Storyline` column) to demonstrate the required NLP step:
1. Lowercase
2. Strip punctuation/non-alphabetic characters
3. Tokenize
4. Remove English stopwords (NLTK stopword list)

Its output is cross-checked against the dataset's own pre-cleaned column as a sanity check (~76% first-5-token agreement — differences come from minor stopword-list and tokenization variations, not an error).

## Text Representation & Similarity

- **TF-IDF Vectorizer** (unigrams + bigrams, top 10,000 features by frequency) converts cleaned storylines into numerical vectors.
- **Cosine Similarity** measures how close a query is to each movie's storyline vector.

**Why the full movie-to-movie similarity matrix is NOT precomputed:** an earlier version of this pipeline did precompute the full 5099×5099 similarity matrix, but at ~100MB (float32) it sat right at GitHub's file-size ceiling for no real benefit — computing similarity for *one* query against the 5099×10000 TF-IDF matrix is a single sparse matrix-vector product (milliseconds), so there's nothing to gain from precomputing all ~26M pairwise scores when only one row is ever needed per request. This is the "Optimized Queries" approach used instead.

## Recommendation Methodology

Two complementary modes, both in `scripts/recommend.py`:

1. **From a typed storyline** (the brief's primary use case): the user describes a plot in their own words → it's cleaned with the same NLP pipeline → vectorized with the fitted TF-IDF vectorizer → compared via cosine similarity against all 5,099 movies → top 5 returned.
2. **From an existing movie**: pick a movie you already know → see others with the most similar storyline (excludes the movie itself).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

## Workflow / Execution

```bash
cd scripts

# 1. Clean and tokenize storylines
python 01_clean_storylines.py --input ../data/imdb_movies_2024_raw.csv --output ../data/cleaned_movies.csv

# 2. Build TF-IDF vectorizer + matrix
python 02_build_similarity_model.py --input ../data/cleaned_movies.csv --output-dir ../models
```

Then launch the app:

```bash
cd ../streamlit_app
streamlit run app.py
```

## App Features

**Tab 1 — Describe a Storyline:** type any plot description → get the top 5 most similar 2024 movies with match scores.

**Tab 2 — Similar to a Movie:** pick a movie from the dataset → see the top 5 most similar other movies.

## Example (from the brief)

Input: *"A young wizard begins his journey at a magical school where he makes friends and enemies, facing dark forces along the way."*

Output (actual result from this dataset): the closest matches are the movies whose storylines share the most thematic/lexical overlap — e.g. a film described as involving "dark forces" reaching out — reflecting genuine TF-IDF similarity rather than fabricated results. Because this dataset covers real, unrelated 2024 releases (not a fantasy-franchise catalog), similarity scores for a distinctive fantasy plot are modest (~20–25%) rather than near-perfect matches — this is expected and correct behavior for a content-based recommender on this data.

## Key Insights

- 5,099 movies from IMDb's 2024 list, zero missing storylines.
- 40 shared titles among genuinely distinct films — a good reminder that "duplicate row" checks need to look at more than just the name column.
- TF-IDF with bigrams (not just unigrams) captures short phrase-level similarity (e.g. "dark forces", "coming of age") better than unigrams alone.

## Tech Stack

- **Languages:** Python 3
- **Libraries:** pandas, NLTK (stopwords), scikit-learn (TfidfVectorizer, cosine_similarity), SciPy (sparse matrices)
- **Visualization/App:** Streamlit
- **Scraping (optional/documented, not executed):** Selenium

## Coding Standards

- PEP 8 compliant, modular scripts (one responsibility per script)
- Shared recommendation logic lives once in `scripts/recommend.py`, imported by both the CLI smoke test and the Streamlit app
- All pipeline scripts are CLI-driven with `argparse` for reproducibility
