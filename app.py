import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from scipy import sparse as sp
import ast
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Moctale - Classic Movies Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Awesome UI ---
st.markdown("""
    <style>
    /* General Body Styles */
    body {
        background-color: #f8f9fc;
    }
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Title Style */
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
    }
    .stMarkdown p {
        font-family: 'Arial', sans-serif;
        text-align: center;
    }

    /* Sidebar Styles */
    section[data-testid="stSidebar"] {
        background-color: #2c2f38;   /* dark slate */
        color: #ffffff;              /* white text */
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    .stRadio label {
        color: #ddd !important;
    }

    /* Movie Card Styles */
    .movie-card {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        text-align: center;
    }
    .movie-card:hover {
        transform: scale(1.05);
    }
    .movie-card img {
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .movie-card .title {
        font-weight: bold;
        color: #333;
        font-size: 1rem;
    }
    .movie-card .caption {
        font-size: 0.8rem;
        color: #666;
    }

    /* Watermark */
    .watermark {
        margin-top: 30px;
        text-align: center;
        color: #aaa;
        font-size: 13px;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)



# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    """Fetches a movie poster URL from The Movie Database (TMDB) API with caching and timeouts."""
    try:
        api_key = st.secrets.get("TMDB_API_KEY", None)
        if not api_key:
            return "https://placehold.co/500x750/cccccc/FFFFFF?text=Missing+TMDB+Key"
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        response = requests.get(url, timeout=(3, 10))
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            return "https://placehold.co/500x750/cccccc/FFFFFF?text=No+Poster"
    except requests.exceptions.RequestException:
        return "https://placehold.co/500x750/ff0000/FFFFFF?text=API+Error"

@st.cache_data
def load_and_prepare_data():
    try:
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        # Prefer joining on stable IDs instead of potentially ambiguous titles
        if 'id' in movies.columns and 'movie_id' in credits.columns:
            movies = movies.merge(credits, left_on='id', right_on='movie_id', how='inner', suffixes=('_m', '_c'))
            # Standardize downstream to use 'movie_id' everywhere
            if 'movie_id' not in movies.columns and 'id' in movies.columns:
                movies['movie_id'] = movies['id']
        else:
            # Fallback to title join if IDs are missing
            movies = movies.merge(credits, on='title', how='inner', suffixes=('_m', '_c'))
            if 'movie_id' not in movies.columns and 'id' in movies.columns:
                movies['movie_id'] = movies['id']

        # Coalesce possible duplicate columns from merge (e.g., title_m/title_c -> title)
        def coalesce(df, base):
            a, b = f"{base}_m", f"{base}_c"
            if a in df.columns or b in df.columns:
                df[base] = df.get(a, pd.Series(index=df.index))
                if b in df.columns:
                    df[base] = df[base].fillna(df[b])
            return df

        for col in ['title', 'overview', 'genres', 'keywords']:
            movies = coalesce(movies, col)

        movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
        movies = movies[movies['release_date'].dt.year < 2000].copy()
        # Keep key metrics for hybrid scoring (ensure we reference unified columns)
        keep_cols = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date', 'vote_average', 'vote_count', 'popularity']
        # Some columns from credits may be named exactly 'cast'/'crew'; keep as is
        movies = movies[[c for c in keep_cols if c in movies.columns]].copy()
        movies.dropna(inplace=True)
        movies.drop_duplicates(inplace=True)
        return movies
    except FileNotFoundError:
        st.error("Dataset files not found. Place 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' in the app folder.")
        return None

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

@st.cache_data
def process_features(_movies_df):
    movies_df = _movies_df.copy()
    for feature in ['genres', 'keywords', 'cast', 'crew']:
        movies_df[feature] = movies_df[feature].apply(safe_literal_eval)

    def get_director(obj):
        return [i['name'] for i in obj if i['job'] == 'Director']

    def get_names(obj):
        if isinstance(obj, list):
            return [i['name'] for i in obj]
        return []

    movies_df['director'] = movies_df['crew'].apply(get_director)
    for feature in ['genres', 'keywords', 'cast']:
        movies_df[feature] = movies_df[feature].apply(get_names)

    for feature in ['genres', 'keywords', 'cast', 'director']:
        movies_df[feature] = movies_df[feature].apply(lambda x: [i.replace(" ", "") for i in x])

    movies_df['tags'] = (
        movies_df['overview'].apply(lambda x: x.split() if isinstance(x, str) else []) +
        movies_df['genres'] +
        movies_df['keywords'] +
        movies_df['cast'] +
        movies_df['director']
    )
    movies_df['tags'] = movies_df['tags'].apply(lambda x: " ".join(x))

    new_df = movies_df[['movie_id', 'title', 'tags', 'release_date', 'genres']].copy()
    # Bring forward metrics if present
    for col in ['vote_average', 'vote_count', 'popularity']:
        if col in _movies_df.columns:
            new_df[col] = _movies_df[col].values
    new_df['release_year'] = new_df['release_date'].dt.year
    new_df['genres_str'] = new_df['genres'].apply(lambda x: ', '.join(x))
    # Compute weighted rating (IMDb-style) if votes available
    if 'vote_average' in new_df.columns and 'vote_count' in new_df.columns:
        C = new_df['vote_average'].mean()
        m = np.percentile(new_df['vote_count'], 80)
        v = new_df['vote_count'].astype(float)
        R = new_df['vote_average'].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            wr = (v/(v+m))*R + (m/(v+m))*C
        new_df['weighted_rating'] = np.where(np.isfinite(wr), wr, R)
        # Normalize metrics to 0-1 for scoring
        new_df['rating_norm'] = (new_df['weighted_rating'] - new_df['weighted_rating'].min()) / (new_df['weighted_rating'].max() - new_df['weighted_rating'].min() + 1e-9)
    else:
        new_df['rating_norm'] = 0.0
    if 'popularity' in new_df.columns:
        pop = new_df['popularity'].astype(float)
        new_df['popularity_norm'] = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
    else:
        new_df['popularity_norm'] = 0.0
    return new_df

@st.cache_resource
def build_tfidf_index(tags):
    """Fit a sparse TF-IDF index for tags and return vectorizer and matrix."""
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(tags)  # keep sparse
    return tfidf, vectors

def recommend(movie, movies_df, vectors, num_recommendations=10):
    try:
        movie_index_label = movies_df[movies_df['title'] == movie].index[0]
        movie_positional_index = movies_df.index.get_loc(movie_index_label)
        # Compute cosine similarity for the selected item against all others on demand
        distances = cosine_similarity(vectors[movie_positional_index], vectors).ravel()
        # Exclude the movie itself and take top N
        top_idx = np.argpartition(-distances, range(1, num_recommendations + 1))[1:num_recommendations + 1]
        # Sort the selected top indices by actual score descending
        top_idx = top_idx[np.argsort(-distances[top_idx])]
        recommended_movies = []
        for idx in top_idx:
            movie_id = movies_df.iloc[idx].movie_id
            title = movies_df.iloc[idx].title
            year = movies_df.iloc[idx].release_year
            genres = movies_df.iloc[idx].genres_str
            poster_url = fetch_poster(movie_id)
            # Build a brief explanation: top overlapping words between tags
            try:
                base_tokens = set(str(movies_df.loc[movie_index_label, 'tags']).split())
                rec_tokens = set(str(movies_df.iloc[idx].tags).split())
                overlap = [w for w in rec_tokens.intersection(base_tokens) if len(w) > 3]
                why = ", ".join(overlap[:5]) if overlap else "Similar theme"
            except Exception:
                why = "Similar theme"
            recommended_movies.append({'id': movie_id, 'title': title, 'year': year, 'genres': genres, 'poster': poster_url, 'why': why})
        return recommended_movies
    except (IndexError, KeyError):
        return []

def build_query_vector(selected_titles, query_text, movies_df, tfidf, vectors):
    """Create a query vector from selected movies and optional text (keywords/people)."""
    parts = []
    # Seed by selected movies
    if selected_titles:
        idxs = []
        for t in selected_titles:
            try:
                label = movies_df[movies_df['title'] == t].index[0]
                idxs.append(movies_df.index.get_loc(label))
            except Exception:
                pass
        if idxs:
            seed_vec = vectors[idxs].mean(axis=0)
            # ensure sparse CSR matrix
            if not sp.issparse(seed_vec):
                seed_vec = sp.csr_matrix(seed_vec)
            else:
                seed_vec = seed_vec.tocsr()
            parts.append(seed_vec)
    # Seed by free-text query
    if query_text and query_text.strip():
        q_vec = tfidf.transform([query_text.strip()])
        parts.append(q_vec)
    if not parts:
        return None
    # Average all parts (keep sparse)
    mixed = parts[0]
    for p in parts[1:]:
        mixed = mixed + p
    mixed = mixed / len(parts)
    return mixed.tocsr()

def recommend_hybrid(selected_titles, movies_df, tfidf, vectors, num_recommendations, query_text, weight_sim, weight_rating, weight_pop):
    q = build_query_vector(selected_titles, query_text, movies_df, tfidf, vectors)
    if q is None:
        return []
    sim = cosine_similarity(q, vectors).ravel()
    # Normalize weights
    w_sum = max(weight_sim + weight_rating + weight_pop, 1e-9)
    ws = weight_sim / w_sum
    wr = weight_rating / w_sum
    wp = weight_pop / w_sum
    rating = movies_df['rating_norm'].values
    pop = movies_df['popularity_norm'].values
    final_score = ws * sim + wr * rating + wp * pop
    # Exclude any selected items from results
    exclude_idx = set()
    for t in selected_titles:
        try:
            label = movies_df[movies_df['title'] == t].index[0]
            exclude_idx.add(movies_df.index.get_loc(label))
        except Exception:
            pass
    order = np.argsort(-final_score)
    ranked = [i for i in order if i not in exclude_idx][:num_recommendations]
    recs = []
    # For overlap explanation, if a single seed is chosen, use it; else use top tokens from query
    base_tokens = set()
    if len(selected_titles) == 1:
        try:
            label = movies_df[movies_df['title'] == selected_titles[0]].index[0]
            base_tokens = set(str(movies_df.loc[label, 'tags']).split())
        except Exception:
            base_tokens = set()
    else:
        base_tokens = set(str(query_text).split()) if query_text else set()
    for idx in ranked:
        movie_id = movies_df.iloc[idx].movie_id
        title = movies_df.iloc[idx].title
        year = movies_df.iloc[idx].release_year
        genres = movies_df.iloc[idx].genres_str
        poster_url = fetch_poster(movie_id)
        try:
            rec_tokens = set(str(movies_df.iloc[idx].tags).split())
            overlap = [w for w in rec_tokens.intersection(base_tokens) if len(w) > 3]
            why = ", ".join(overlap[:5]) if overlap else "Balanced score"
        except Exception:
            why = "Balanced score"
        recs.append({'id': movie_id, 'title': title, 'year': year, 'genres': genres, 'poster': poster_url, 'why': why})
    return recs

# --- ML Model Integration ---
def rerank_with_ml(recommendations):
    try:
        clf = joblib.load("ml_model.pkl")
        mlb = joblib.load("mlb.pkl")
    except:
        return recommendations  # fallback

    ranked = []
    for movie in recommendations:
        genres = movie['genres'].split(", ")
        try:
            X = mlb.transform([genres])
            score = clf.predict_proba(X)[0][1]
        except:
            score = 0
        ranked.append((movie, score))
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    return [m for m, _ in ranked]


# --- Main Application ---
def main():
    with st.sidebar:
        st.title("🎬 Moctale")
        st.subheader("Classic Movies Recommender")
        st.markdown("Discover timeless classics tailored to your taste.")
        
        # Sidebar options
        st.markdown("---")
        algo_choice = st.radio("Recommendation Mode", ["Content-Based", "ML Enhanced", "Hybrid (Beast Mode)"])
        num_recs = st.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=1)
        # Decade filter
        decade_options = ["All", "1920s", "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s"]
        selected_decade = st.selectbox("Filter by decade", options=decade_options, index=0)
        # Hybrid controls
        preset = st.selectbox("Discovery preset", ["Custom", "Hidden gems", "Critically acclaimed", "Popular"], index=0)
        weight_sim = st.slider("Weight: Similarity", 0.0, 1.0, 0.6, 0.05)
        weight_rating = st.slider("Weight: Rating", 0.0, 1.0, 0.3, 0.05)
        weight_pop = st.slider("Weight: Popularity", 0.0, 1.0, 0.1, 0.05)
        if preset == "Hidden gems":
            weight_sim, weight_rating, weight_pop = 0.6, 0.35, 0.05
        elif preset == "Critically acclaimed":
            weight_sim, weight_rating, weight_pop = 0.4, 0.55, 0.05
        elif preset == "Popular":
            weight_sim, weight_rating, weight_pop = 0.4, 0.2, 0.4
        st.markdown("---")
        with st.expander("ℹ️ About"):
            st.markdown("""
            **How it works:**  
            - *Content-Based*: Uses movie metadata (genres, cast, overview) and cosine similarity.  
            - *ML Enhanced*: Re-ranks results with a trained RandomForest model.  
            """)
        with st.expander("👨‍💻 Credits"):
            st.markdown("Developed with ❤️ by **Suyash Mukherjee**")

    st.title("Moctale - Classic Movies Recommender System")
    st.markdown("<p>Find your next favorite classic film!</p>", unsafe_allow_html=True)

    # Watermark at bottom of main screen
    st.markdown('<div class="watermark">Developed & Maintained by Suyash Mukherjee</div>', unsafe_allow_html=True)

    movies_data = load_and_prepare_data()
    if movies_data is not None:
        processed_movies = process_features(movies_data)
        # Apply decade filter before building vectors
        if selected_decade != "All":
            decade_start = int(selected_decade[:4])
            decade_end = decade_start + 9
            processed_movies = processed_movies[
                (processed_movies['release_year'] >= decade_start) & (processed_movies['release_year'] <= decade_end)
            ]
        tfidf, vectors = build_tfidf_index(processed_movies['tags'])

        # Dataset Insights
        with st.expander("📊 Dataset Insights", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Movies (pre-2000)", int(len(processed_movies)))
                st.metric("Distinct genres", int(processed_movies['genres_str'].str.split(', ').explode().nunique()))
            with col2:
                decade_series = (processed_movies['release_year'] // 10 * 10).value_counts().sort_index()
                st.bar_chart(decade_series.rename_axis('Decade'))
            with col3:
                top_genres = processed_movies['genres_str'].str.split(', ').explode().value_counts().head(10)
                st.write("Top genres:")
                st.table(top_genres.to_frame('Count'))

        movie_list = sorted(processed_movies['title'].unique())
        selected_movies = st.multiselect(
            "🎥 Choose one or more classic movies (multi-seed):",
            options=movie_list,
            default=[]
        )
        query_keywords = st.text_input("Optional: keywords, actors, directors to guide results", value="")

        if selected_movies or query_keywords.strip():
            st.markdown("---")
            if selected_movies:
                st.subheader(f"Top {num_recs} Recommendations")
            else:
                st.subheader(f"Top {num_recs} Recommendations (guided by keywords)")

            with st.spinner('🎬 Finding the best classics for you...'):
                if algo_choice == "Hybrid (Beast Mode)":
                    recommendations = recommend_hybrid(selected_movies, processed_movies, tfidf, vectors, num_recommendations=num_recs, query_text=query_keywords, weight_sim=weight_sim, weight_rating=weight_rating, weight_pop=weight_pop)
                else:
                    # Fall back to single-seed content-based using first selection
                    seed_title = selected_movies[0] if selected_movies else None
                    recommendations = recommend(seed_title, processed_movies, vectors, num_recommendations=num_recs) if seed_title else []
                if algo_choice == "ML Enhanced":
                    recommendations = rerank_with_ml(recommendations)

            if recommendations:
                cols = st.columns(5)
                for i, movie in enumerate(recommendations):
                    with cols[i % 5]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <img src="{movie['poster']}" width="100%">
                            <div class="title">{movie['title']} ({movie['year']})</div>
                            <div class="caption">Genres: {movie['genres']}</div>
                            <div class="caption">Why: {movie.get('why', 'Similar theme')}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Could not generate recommendations. Try another movie.")
    else:
        st.error("Application cannot start due to data loading issues.")

if __name__ == '__main__':
    main()


