"""
IMDB Movie Recommendation System Using Storylines
Streamlit application implementing the project brief's recommendation engine.

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import recommend as rec  # noqa: E402

st.set_page_config(page_title="IMDB Movie Recommender", page_icon="🎬", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@st.cache_data
def load_data():
    return rec.load_artifacts(
        os.path.join(DATA_DIR, "cleaned_movies.csv"),
        os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
        os.path.join(MODELS_DIR, "tfidf_matrix.npz"),
    )


movies, vectorizer, tfidf_matrix = load_data()

st.title("🎬 IMDB Movie Recommendation System")
st.caption(f"Storyline-based recommendations across {len(movies):,} movies from IMDb's 2024 list")

tab1, tab2 = st.tabs(["✍️ Describe a Storyline", "🎞️ Similar to a Movie"])

# ---------------------------------------------------------------------------
# Tab 1: user types a storyline -> get top 5 similar movies
# ---------------------------------------------------------------------------

with tab1:
    st.markdown("Describe a plot, and we'll find the top 5 most similar 2024 movies.")

    example = "A young wizard begins his journey at a magical school where he makes friends and enemies, facing dark forces along the way."
    user_input = st.text_area("Movie storyline", placeholder=example, height=100)

    top_n = st.slider("Number of recommendations", 3, 10, 5, key="storyline_n")

    if st.button("Get Recommendations", type="primary"):
        if not user_input.strip():
            st.warning("Type a storyline description to get recommendations.")
        else:
            with st.spinner("Finding similar movies..."):
                results = rec.recommend_from_storyline(user_input, movies, vectorizer, tfidf_matrix, top_n=top_n)

            if results.empty or results["similarity"].max() == 0:
                st.info("No meaningfully similar movies found — try a more descriptive storyline.")
            else:
                st.success(f"Top {len(results)} similar movies")
                for _, row in results.iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(f"**{row['movie_name']}**")
                            st.caption(row["storyline"])
                        with c2:
                            st.metric("Match", f"{row['similarity']:.0%}")

# ---------------------------------------------------------------------------
# Tab 2: pick an existing movie -> get top 5 similar movies
# ---------------------------------------------------------------------------

with tab2:
    st.markdown("Pick a movie you like, and we'll find others with a similar storyline.")

    movie_choice = st.selectbox("Choose a movie", sorted(movies["movie_name"].unique()))
    top_n2 = st.slider("Number of recommendations", 3, 10, 5, key="movie_n")

    if st.button("Find Similar Movies", type="primary"):
        selected_storyline = movies[movies["movie_name"] == movie_choice]["storyline"].iloc[0]
        st.caption(f"**{movie_choice}**: {selected_storyline}")

        with st.spinner("Finding similar movies..."):
            results = rec.recommend_similar_to_movie(movie_choice, movies, tfidf_matrix, top_n=top_n2)

        if results.empty:
            st.info("No similar movies found.")
        else:
            st.success(f"Top {len(results)} movies similar to {movie_choice}")
            for _, row in results.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**{row['movie_name']}**")
                        st.caption(row["storyline"])
                    with c2:
                        st.metric("Match", f"{row['similarity']:.0%}")
