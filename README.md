# 🎬 BERTopic Movie Clustering & Recommendation System

A modern topic-modeling and neural recommendation engine built with **BERTopic**, **Sentence-Transformers (`all-MiniLM-L6-v2`)**, **UMAP**, **HDBSCAN**, and **Class-based TF-IDF (c-TF-IDF)** on 9,737 movies.

---

## 🌟 Key Features

1. **Neural Topic Modeling (BERTopic)**:
   - Encodes rich movie synopses (Title, Year, Genres, Director, Cast, and Full Plot) into 384-dimensional dense semantic vectors.
   - Preserves non-linear manifold geometry using UMAP.
   - Automatically discovers hierarchical semantic topics using density-based HDBSCAN.
   - Extracts human-interpretable topic keywords using Class-based TF-IDF (c-TF-IDF).

2. **Movie-to-Movie Recommendations**:
   - Recommends films based on deep plot similarity, stylistic tone, shared personnel, and semantic topic alignment.
   - Explains *why* each movie was recommended (shared topic, shared genres, common director/cast, high IMDb acclaim).

3. **Personalized Recommendations for People (Taste Profiles)**:
   - Aggregates multiple liked films into a user taste centroid vector:
     $$\vec{u} = \frac{1}{|M|} \sum_{m \in M} \vec{e}_m$$
   - Discovers the user's primary topic affinities (e.g. *Space Sci-Fi*, *Noir Crime*, *Animated Family*).
   - Ranks unseen movies combining cosine similarity, topic match, and Bayesian weighted rating ($WR$).

4. **Zero-Shot Natural Language Mood Search**:
   - Allows users to describe in plain English what they feel like watching (e.g., *"creepy haunted house and demonic paranormal investigation"* or *"mind-bending space sci-fi where scientists question reality"*).
   - Semantically retrieves the top matching films instantly.

5. **2D Interactive Topic Galaxy**:
   - Visualizes all 9,700+ movies in an interactive Plotly map colored by topic.

6. **Instant Caching**:
   - All 9,737 neural embeddings and topic models are cached in `data/`, allowing instant startup in **< 2 seconds**.

---

## 🚀 Quick Start

### 1. Launch the Interactive Web App (Streamlit)
Double-click `run_app.bat` or run:
```bash
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 2. Launch the Interactive Command Line Interface (CLI)
```bash
python recommender_cli.py
```
Select from:
- `[1]` Recommend by movie title
- `[2]` Build personalized user profile
- `[3]` Natural language mood search
- `[4]` Explore discovered topics
- `[5]` Search movie database

---

## 🧠 Algorithmic Pipeline

```
Raw Movie Data (Title, Plot, Cast, Director, Genre)
                       │
                       ▼
  Sentence-Transformers (all-MiniLM-L6-v2, 384D)
                       │
                       ▼
       UMAP (Dimensionality Reduction)
                       │
                       ▼
      HDBSCAN (Density-Based Clustering)
                       │
                       ▼
c-TF-IDF (Class-based Topic Keyword Extraction)
                       │
                       ▼
    Multi-Factor Recommendation Scoring
  (Semantic Cosine + Topic Boost + Bayesian IMDb Quality)
```

### Recommendation Scoring Formula
For candidate movie $i$ given target vector $\vec{v}$:
$$\text{Score}_i = 0.70 \cdot \cos(\vec{v}, \vec{e}_i) + 0.15 \cdot \mathbb{I}(T_i = T_{\text{target}}) + 0.15 \cdot Q_i$$
where:
- $\cos(\vec{v}, \vec{e}_i)$ is normalized dot product in 384-dimensional embedding space.
- $\mathbb{I}(T_i = T_{\text{target}})$ is the topic affinity bonus.
- $Q_i$ is the min-max normalized Bayesian weighted rating:
  $$WR_i = \frac{v_i}{v_i + m} R_i + \frac{m}{v_i + m} C$$

---

## 💻 Python API Usage

```python
from movie_recommender import MovieRecommenderSystem

# Initialize and load cached engine (loads in ~2 seconds)
engine = MovieRecommenderSystem(csv_path="data/movies.csv", min_topic_size=15)
engine.fit(force_retrain=False)

# 1. Movie-to-Movie
res = engine.recommend_by_movie("Inception", top_n=5)
for rec in res["recommendations"]:
    print(f"#{rec['rank']} {rec['title']} ({rec['year']}) - Match: {rec['composite_score']}")

# 2. Personalized User Profile
user_res = engine.recommend_for_user(
    liked_movies=["Toy Story", "Finding Nemo"],
    preferred_genres=["Animation", "Adventure"],
    top_n=5
)
for rec in user_res["recommendations"]:
    print(f"#{rec['rank']} {rec['title']} - Score: {rec['match_score']}")

# 3. Natural Language Mood Search
mood_res = engine.recommend_by_mood_or_query(
    "creepy haunted house and demonic paranormal investigation",
    top_n=5
)
for rec in mood_res["recommendations"]:
    print(f"#{rec['rank']} {rec['title']} - Semantic: {rec['semantic_score']}")
```

---

## 📁 File Structure

```
├── app.py                         # Streamlit interactive web dashboard
├── recommender_cli.py             # Interactive terminal tool
├── run_app.bat                    # 1-click Windows runner
├── train_and_test_bertopic.py     # Evaluation & test suite
├── evaluate_clusters.py           # Quantitative metric evaluation (Elbow, Silhouette, DB)
├── README_BERTOPIC_RECOMMENDER.md # This documentation
├── movie_recommender/
│   ├── __init__.py
│   ├── preprocessor.py            # Metadata cleaning & document formulation
│   ├── bertopic_model.py          # BERTopic pipeline, embeddings, UMAP, c-TF-IDF
│   └── engine.py                  # Recommender scoring, user profile & search
└── data/
    ├── movies.csv                 # Source dataset (9,742 movies)
    ├── movies_processed.csv       # Cleaned movies with assigned topics & 2D coords
    ├── movie_embeddings.npy       # Precomputed 384D Sentence-BERT embeddings
    ├── coords_2d.npy              # 2D UMAP projection coordinates
    └── bertopic_model/            # Serialized BERTopic model
```
