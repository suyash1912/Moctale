**Try the app here:** 
```bash
https://moctale.streamlit.app/
```

---
#  Moctale: A Classic Movie Recommender System

Moctale is a **Streamlit-based application** designed to recommend **classic movies released before the year 2000**.  
It offers multiple recommendation strategies, from simple content-based filtering to a sophisticated hybrid model that blends content similarity with movie popularity and ratings.

---

##  Features

- **Content-Based Filtering** → Recommends movies based on similarity of genres, keywords, cast, and crew.  
- **ML Enhanced Reranking** → Uses a trained Random Forest model to re-rank recommendations, prioritizing movies predicted to be more popular.  
- **Hybrid "Beast Mode"** → Combines content similarity with weighted ratings and popularity for flexible discovery of *hidden gems* or *critically acclaimed* films.  
- **Customizable Recommendations** → Adjust similarity, rating, and popularity weights to fine-tune results.  
- **Decade-Based Filtering** → Narrow recommendations to specific decades.  
- **Multi-Seed Recommendations** → Get suggestions based on multiple movies and custom keywords.  

---

##  How It Works

The system runs on two main scripts:

- **`app.py`** → Builds the Streamlit interface, handles data loading, feature processing, and runs different recommendation algorithms.  
  - Uses **TF-IDF Vectorizer** + **Cosine Similarity** for content-based matching.  
  - Integrates with the **TMDB API** to fetch posters.  

- **`model.py`** → Trains a **Random Forest Classifier** to predict movie popularity based on genres.  
  - Saves the trained model as `ml_model.pkl` (along with `mlb.pkl`).  
  - Used in `app.py` for ML-enhanced re-ranking.  

---

## Data Source

This app uses datasets from **The Movie Database (TMDB)**:

- `tmdb_5000_movies.csv`  
- `tmdb_5000_credits.csv`  

Both files must be in the same directory as the app scripts for proper functionality.

---

## Setup & Installation

###  Prerequisites
- Python **3.7+**

###  Steps

1. **Clone the Repository**
```bash
 git clone https://github.com/your-username/Moctale.git
 cd Moctale
```
2. **Install Required Packages**
```bash
pip install -r requirements.txt
 ```
3. **Obtain a TMDB API Key**

Create a free account at TMDB.
Go to Settings → API and generate a new API key.
  
4. **Create secrets.toml**

Inside your project folder, create:
.streamlit/secrets.toml

Add your API key:
```bash
TMDB_API_KEY = "YOUR_API_KEY_HERE"
 ```
5. **Run the streamlit server**
```bash
streamlit run app.py
 ```
---

**Contributing**

Contributions are welcome! 🎉
Feel free to open issues or submit pull requests to improve Moctale.
     
