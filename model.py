import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import ast

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

def train_ml_model():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")

    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies = movies[movies['release_date'].dt.year < 2000].copy()
    movies.dropna(subset=["genres"], inplace=True)

    movies['genres'] = movies['genres'].apply(safe_literal_eval)
    movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(movies['genres'])

    X = genre_features
    y = (movies['popularity'] > movies['popularity'].median()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # --- Model Checking (Accuracy) ---
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Test Accuracy: {acc:.2f}")

    joblib.dump(clf, "ml_model.pkl")
    joblib.dump(mlb, "mlb.pkl")

if __name__ == "__main__":
    train_ml_model()

