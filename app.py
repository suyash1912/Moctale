import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
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

def fetch_poster(movie_id):
    """Fetches a movie poster URL from The Movie Database (TMDB) API."""
    try:
        api_key = "e447ab3a21e9a3776d39b9b139aba828"  # Replace with your TMDB API key
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        response = requests.get(url)
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
        movies = movies.merge(credits, on='title')
        movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
        movies = movies[movies['release_date'].dt.year < 2000].copy()
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date']]
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
    new_df['release_year'] = new_df['release_date'].dt.year
    new_df['genres_str'] = new_df['genres'].apply(lambda x: ', '.join(x))
    return new_df

@st.cache_resource
def get_similarity_matrix(tags):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(tags).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(movie, movies_df, similarity_matrix, num_recommendations=10):
    try:
        movie_index_label = movies_df[movies_df['title'] == movie].index[0]
        movie_positional_index = movies_df.index.get_loc(movie_index_label)
        distances = similarity_matrix[movie_positional_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
        recommended_movies = []
        for i in movies_list:
            movie_id = movies_df.iloc[i[0]].movie_id
            title = movies_df.iloc[i[0]].title
            year = movies_df.iloc[i[0]].release_year
            genres = movies_df.iloc[i[0]].genres_str
            poster_url = fetch_poster(movie_id)
            recommended_movies.append({'id': movie_id, 'title': title, 'year': year, 'genres': genres, 'poster': poster_url})
        return recommended_movies
    except (IndexError, KeyError):
        return []

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
        algo_choice = st.radio("Recommendation Mode", ["Content-Based", "ML Enhanced"])
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
        similarity = get_similarity_matrix(processed_movies['tags'])

        movie_list = sorted(processed_movies['title'].unique())
        selected_movie = st.selectbox(
            "🎥 Choose a classic movie:",
            options=movie_list,
            index=None,
            placeholder="Type or select a movie..."
        )

        if selected_movie:
            st.markdown("---")
            st.subheader(f"Top 10 Recommendations for '{selected_movie}'")

            with st.spinner('🎬 Finding the best classics for you...'):
                recommendations = recommend(selected_movie, processed_movies, similarity)
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
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Could not generate recommendations. Try another movie.")
    else:
        st.error("Application cannot start due to data loading issues.")

if __name__ == '__main__':
    main()

