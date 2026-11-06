from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
import pandas as pd
from .preprocessor import MovieDataPreprocessor
from .bertopic_model import MovieBERTopic


class MovieRecommenderSystem:
    """BERTopic-Powered Semantic Movie Topic Modeling & Recommendation Engine."""

    def __init__(
        self,
        csv_path: str = "data/movies.csv",
        min_topic_size: int = 15,
        cache_dir: str = "data",
    ):
        self.csv_path = csv_path
        self.preprocessor = MovieDataPreprocessor(csv_path)
        self.topic_model = MovieBERTopic(min_topic_size=min_topic_size, cache_dir=cache_dir)
        self.df = None
        self.embeddings = None
        self.is_fitted = False

    def fit(self, force_retrain: bool = False):
        """Loads data, prepares rich movie documents, and trains/loads BERTopic."""
        self.df = self.preprocessor.load_data()
        self.df, self.embeddings = self.topic_model.fit_transform(
            self.df, force_retrain=force_retrain
        )
        self.is_fitted = True
        return self

    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Finds movies matching query string."""
        if not self.is_fitted:
            raise RuntimeError("Recommender must be fitted before searching.")
        q = query.strip().lower()
        if not q:
            return pd.DataFrame()

        exact = self.df[self.df["Title"].str.lower() == q]
        starts = self.df[
            self.df["Title"].str.lower().str.startswith(q)
            & (self.df["Title"].str.lower() != q)
        ]
        contains = self.df[
            self.df["Title"].str.lower().str.contains(q, regex=False)
            & (~self.df.index.isin(exact.index))
            & (~self.df.index.isin(starts.index))
        ]

        results = pd.concat([exact, starts, contains]).head(limit)
        cols = ["Title", "Year", "clean_rating", "Genre", "Director", "topic"]
        available_cols = [c for c in cols if c in results.columns]
        return results[available_cols]

    def _get_movie_index(self, identifier) -> Optional[int]:
        if isinstance(identifier, int) and 0 <= identifier < len(self.df):
            return identifier
        query = str(identifier).strip().lower()
        matches = self.df[self.df["Title"].str.lower() == query]
        if not matches.empty:
            return matches.index[0]
        matches = self.df[self.df["Title"].str.lower().str.contains(query, regex=False)]
        if not matches.empty:
            return matches.index[0]
        return None

    def explain_similarity(self, target_idx: int, cand_idx: int) -> List[str]:
        """Provides natural language explanation for why candidate is recommended."""
        target = self.df.iloc[target_idx]
        cand = self.df.iloc[cand_idx]
        reasons = []

        # Shared BERTopic
        if target.get("topic") == cand.get("topic") and target.get("topic") != -1:
            topic_name = self.topic_model.get_topic_name(int(target["topic"]))
            reasons.append(f"Same semantic topic: {topic_name}")

        # Shared genres
        t_genres = {g.strip().lower() for g in str(target["Genre"]).split(",") if g.strip()}
        c_genres = {g.strip().lower() for g in str(cand["Genre"]).split(",") if g.strip()}
        shared_genres = t_genres.intersection(c_genres)
        if shared_genres:
            reasons.append(f"Shared genres: {', '.join([g.title() for g in shared_genres])}")

        # Shared director
        if target["Director"] and cand["Director"] and target["Director"] != "N/A":
            t_dir = {d.strip().lower() for d in str(target["Director"]).split(",")}
            c_dir = {d.strip().lower() for d in str(cand["Director"]).split(",")}
            shared_dir = t_dir.intersection(c_dir)
            if shared_dir:
                reasons.append(f"Same director: {', '.join([d.title() for d in shared_dir])}")

        # Shared cast
        if target["Actors"] and cand["Actors"] and target["Actors"] != "N/A":
            t_act = {a.strip().lower() for a in str(target["Actors"]).split(",")}
            c_act = {a.strip().lower() for a in str(cand["Actors"]).split(",")}
            shared_act = t_act.intersection(c_act)
            if shared_act:
                reasons.append(f"Common cast: {', '.join([a.title() for a in shared_act])}")

        # Rating badge
        if cand["clean_rating"] >= 8.0:
            reasons.append(f"Critically acclaimed (IMDb {cand['clean_rating']}/10)")

        return reasons if reasons else ["High neural plot & style similarity"]

    def recommend_by_movie(
        self,
        movie_query: str,
        top_n: int = 6,
        topic_boost: float = 0.15,
        quality_weight: float = 0.15,
        min_rating: float = 0.0,
    ) -> Dict[str, Any]:
        """Recommends movies similar to a target movie using BERTopic embeddings and topics."""
        if not self.is_fitted:
            raise RuntimeError("Recommender must be fitted before generating recommendations.")

        idx = self._get_movie_index(movie_query)
        if idx is None:
            return {"error": f"Movie '{movie_query}' not found in database."}

        target = self.df.iloc[idx]
        target_topic = int(target.get("topic", -1))
        target_vec = self.embeddings[idx]

        # Cosine similarity in normalized embedding space
        sims = np.dot(self.embeddings, target_vec)

        # Topic match bonus
        topic_bonus = (self.df["topic"] == target_topic).astype(float) * topic_boost

        # Quality component
        quality_component = self.df["quality_score"].values * quality_weight

        # Composite score
        scores = (1.0 - topic_boost - quality_weight) * sims + topic_bonus + quality_component

        # Exclude self
        scores[idx] = -1.0

        if min_rating > 0:
            scores[self.df["clean_rating"] < min_rating] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_n]

        recommendations = []
        for rank, cand_idx in enumerate(top_indices, 1):
            row = self.df.iloc[cand_idx]
            reasons = self.explain_similarity(idx, cand_idx)
            cand_topic = int(row.get("topic", -1))
            recommendations.append(
                {
                    "rank": rank,
                    "title": row["Title"],
                    "year": int(row["clean_year"]),
                    "rating": row["clean_rating"],
                    "genre": row["Genre"],
                    "director": row["Director"],
                    "actors": row["Actors"],
                    "plot": row["Plot"],
                    "poster": row["Poster"],
                    "topic_id": cand_topic,
                    "topic_name": self.topic_model.get_topic_name(cand_topic),
                    "similarity_score": round(float(sims[cand_idx]), 3),
                    "composite_score": round(float(scores[cand_idx]), 3),
                    "reasons": reasons,
                }
            )

        return {
            "target_movie": {
                "title": target["Title"],
                "year": int(target["clean_year"]),
                "rating": target["clean_rating"],
                "genre": target["Genre"],
                "director": target["Director"],
                "actors": target["Actors"],
                "plot": target["Plot"],
                "poster": target["Poster"],
                "topic_id": target_topic,
                "topic_name": self.topic_model.get_topic_name(target_topic),
            },
            "recommendations": recommendations,
        }

    def recommend_for_user(
        self,
        liked_movies: List[str],
        disliked_movies: Optional[List[str]] = None,
        preferred_genres: Optional[List[str]] = None,
        top_n: int = 8,
        min_rating: float = 6.0,
    ) -> Dict[str, Any]:
        """Recommends personalized movies for a user taste profile."""
        if not self.is_fitted:
            raise RuntimeError("Recommender must be fitted first.")

        liked_indices = []
        for m in liked_movies:
            idx = self._get_movie_index(m)
            if idx is not None:
                liked_indices.append(idx)

        if not liked_indices and not preferred_genres:
            return {"error": "Please provide at least one movie or genre you like."}

        # User taste vector from embeddings
        vectors = []
        user_topics = []
        for idx in liked_indices:
            vectors.append(self.embeddings[idx])
            t = int(self.df.loc[idx, "topic"])
            if t != -1:
                user_topics.append(t)

        if vectors:
            user_vec = np.mean(vectors, axis=0)
        else:
            user_vec = np.zeros(self.embeddings.shape[1])

        # If genres provided, embed genre query
        if preferred_genres:
            genre_text = "Genres: " + ", ".join(preferred_genres)
            genre_emb = self.topic_model.embed_text(genre_text)
            user_vec = 0.7 * user_vec + 0.3 * genre_emb

        # Disliked movie negative vector
        if disliked_movies:
            for m in disliked_movies:
                d_idx = self._get_movie_index(m)
                if d_idx is not None:
                    user_vec -= 0.3 * self.embeddings[d_idx]

        # Normalize user vector
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec = user_vec / norm

        # Compute cosine similarity
        sims = np.dot(self.embeddings, user_vec)

        # Topic affinity bonus
        topic_bonus = np.zeros(len(self.df))
        top_affinity_names = []
        if user_topics:
            topic_counts = Counter(user_topics)
            top_t = topic_counts.most_common(1)[0][0]
            topic_bonus = (self.df["topic"] == top_t).astype(float) * 0.15
            top_affinity_names = [
                self.topic_model.get_topic_name(t)
                for t, _ in topic_counts.most_common(2)
            ]

        # Score
        scores = 0.7 * sims + 0.15 * topic_bonus + 0.15 * self.df["quality_score"].values

        # Mask seen movies
        for idx in liked_indices:
            scores[idx] = -1.0

        if min_rating > 0:
            scores[self.df["clean_rating"] < min_rating] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_n]

        recs = []
        for rank, cand_idx in enumerate(top_indices, 1):
            row = self.df.iloc[cand_idx]
            cand_topic = int(row.get("topic", -1))
            recs.append(
                {
                    "rank": rank,
                    "title": row["Title"],
                    "year": int(row["clean_year"]),
                    "rating": row["clean_rating"],
                    "genre": row["Genre"],
                    "director": row["Director"],
                    "actors": row["Actors"],
                    "plot": row["Plot"],
                    "poster": row["Poster"],
                    "topic_name": self.topic_model.get_topic_name(cand_topic),
                    "match_score": round(float(scores[cand_idx]), 3),
                }
            )

        return {
            "taste_profile": {
                "liked_movies": [self.df.loc[i, "Title"] for i in liked_indices],
                "top_affinity_topics": top_affinity_names,
            },
            "recommendations": recs,
        }

    def recommend_by_mood_or_query(
        self,
        query: str,
        top_n: int = 8,
        min_rating: float = 6.0,
    ) -> Dict[str, Any]:
        """Zero-shot recommendation: finds movies matching a natural language mood/description."""
        if not self.is_fitted:
            raise RuntimeError("Recommender must be fitted first.")

        query_emb = self.topic_model.embed_text(query)
        sims = np.dot(self.embeddings, query_emb)

        scores = 0.8 * sims + 0.2 * self.df["quality_score"].values
        if min_rating > 0:
            scores[self.df["clean_rating"] < min_rating] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_n]

        recs = []
        for rank, cand_idx in enumerate(top_indices, 1):
            row = self.df.iloc[cand_idx]
            cand_topic = int(row.get("topic", -1))
            recs.append(
                {
                    "rank": rank,
                    "title": row["Title"],
                    "year": int(row["clean_year"]),
                    "rating": row["clean_rating"],
                    "genre": row["Genre"],
                    "director": row["Director"],
                    "plot": row["Plot"],
                    "poster": row["Poster"],
                    "topic_name": self.topic_model.get_topic_name(cand_topic),
                    "semantic_score": round(float(sims[cand_idx]), 3),
                    "match_score": round(float(scores[cand_idx]), 3),
                }
            )

        return {
            "query": query,
            "recommendations": recs,
        }

    def get_topic_summary(self) -> pd.DataFrame:
        """Returns summary dataframe of all discovered BERTopics."""
        rows = []
        unique_topics = sorted([t for t in self.df["topic"].unique() if t != -1])
        for t in unique_topics:
            t_df = self.df[self.df["topic"] == t]
            top_movies = (
                t_df.sort_values(by="weighted_rating", ascending=False)["Title"]
                .head(3)
                .tolist()
            )
            top_genres = (
                t_df["Genre"]
                .str.split(",")
                .explode()
                .str.strip()
                .value_counts()
                .head(2)
                .index.tolist()
            )
            rows.append(
                {
                    "Topic ID": t,
                    "Topic Name": self.topic_model.get_topic_name(t),
                    "Movies Count": len(t_df),
                    "Dominant Genres": ", ".join(top_genres),
                    "Top Rated Movies": ", ".join(top_movies),
                }
            )
        return pd.DataFrame(rows)
