import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize


class MovieBERTopic:
    """Manages the BERTopic model lifecycle, embeddings, topics, and persistence."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_topic_size: int = 15,
        cache_dir: str = "data",
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.min_topic_size = min_topic_size
        self.cache_dir = cache_dir
        self.random_state = random_state

        self.embedding_model = None
        self.topic_model = None
        self.embeddings = None
        self.embeddings_norm = None
        self.coords_2d = None
        self.topics = None
        self.topic_labels_map = {}

    def _get_paths(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        return {
            "embeddings": os.path.join(self.cache_dir, "movie_embeddings.npy"),
            "coords": os.path.join(self.cache_dir, "coords_2d.npy"),
            "df": os.path.join(self.cache_dir, "movies_processed.csv"),
            "model": os.path.join(self.cache_dir, "bertopic_model"),
        }

    def has_cached_model(self) -> bool:
        p = self._get_paths()
        return (
            os.path.exists(p["embeddings"])
            and os.path.exists(p["coords"])
            and os.path.exists(p["df"])
            and os.path.exists(p["model"])
        )

    def load_cached(self, df_preprocessed: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Loads precomputed embeddings and BERTopic model from cache."""
        p = self._get_paths()
        print("Loading cached BERTopic embeddings and model...")
        self.embeddings = np.load(p["embeddings"])
        self.embeddings_norm = normalize(self.embeddings, norm="l2")
        self.coords_2d = np.load(p["coords"])

        # Load processed dataframe containing topics
        cached_df = pd.read_csv(p["df"])
        for col in ["topic", "topic_name", "coord_x", "coord_y"]:
            if col in cached_df.columns:
                df_preprocessed[col] = cached_df[col]

        self.topics = df_preprocessed["topic"].values
        self.embedding_model = SentenceTransformer(self.model_name)

        try:
            self.topic_model = BERTopic.load(p["model"], embedding_model=self.embedding_model)
        except Exception:
            # Recreate lightweight wrapper if serializing custom submodules differed
            self.topic_model = None

        self._build_topic_labels(df_preprocessed)
        return df_preprocessed, self.embeddings_norm

    def fit_transform(
        self, df: pd.DataFrame, force_retrain: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Fits BERTopic pipeline on the movies dataframe or loads cached version."""
        p = self._get_paths()
        if not force_retrain and self.has_cached_model():
            return self.load_cached(df)

        print(f"Initializing SentenceTransformer ({self.model_name})...")
        self.embedding_model = SentenceTransformer(self.model_name)

        docs = df["bertopic_doc"].tolist()
        print(f"Generating neural embeddings for {len(docs)} movies...")
        self.embeddings = self.embedding_model.encode(
            docs, show_progress_bar=True, batch_size=64, normalize_embeddings=False
        )
        self.embeddings_norm = normalize(self.embeddings, norm="l2")

        # 2D projection for visualization
        print("Computing 2D UMAP projection for interactive exploration...")
        umap_2d = UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=self.random_state,
        )
        self.coords_2d = umap_2d.fit_transform(self.embeddings)
        df["coord_x"] = self.coords_2d[:, 0]
        df["coord_y"] = self.coords_2d[:, 1]

        # BERTopic subcomponents
        umap_model = UMAP(
            n_neighbors=15,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        print("Fitting BERTopic (UMAP + HDBSCAN + c-TF-IDF)...")
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            verbose=True,
        )

        topics, probs = self.topic_model.fit_transform(docs, embeddings=self.embeddings)

        # Reduce outliers so all movies receive a topic
        print("Reducing outlier movies (-1) to their closest topic...")
        try:
            topics = self.topic_model.reduce_outliers(
                docs, topics, strategy="embeddings", embeddings=self.embeddings
            )
            self.topic_model.update_topics(docs, topics=topics)
        except Exception as e:
            print(f"Note on outlier reduction: {e}")

        df["topic"] = topics
        self.topics = np.array(topics)

        self._build_topic_labels(df)
        df["topic_name"] = df["topic"].map(self.topic_labels_map)

        # Save to cache
        print("Saving fitted BERTopic model and embeddings to disk...")
        np.save(p["embeddings"], self.embeddings)
        np.save(p["coords"], self.coords_2d)
        df.to_csv(p["df"], index=False)
        try:
            self.topic_model.save(p["model"], serialization="safetensors", save_embedding_model=False)
        except Exception:
            try:
                self.topic_model.save(p["model"], serialization="pickle", save_embedding_model=False)
            except Exception as e:
                print(f"Model save notice: {e}")

        print("[OK] BERTopic training and caching completed!")
        return df, self.embeddings_norm

    def _build_topic_labels(self, df: pd.DataFrame):
        """Generates clean human-readable names for each topic."""
        unique_topics = set(df["topic"].unique())
        for t in unique_topics:
            if t == -1:
                self.topic_labels_map[t] = "Miscellaneous / Outliers"
                continue

            # Fetch top keywords from BERTopic if available
            words = []
            if self.topic_model is not None:
                topic_words = self.topic_model.get_topic(t)
                if topic_words:
                    words = [w[0] for w in topic_words[:3] if len(w[0]) > 2]

            # Fallback to dominant genre in cluster if words empty
            t_df = df[df["topic"] == t]
            top_genre = "Cinema"
            if not t_df.empty and "Genre" in t_df:
                genre_counts = t_df["Genre"].str.split(",").explode().str.strip().value_counts()
                if not genre_counts.empty:
                    top_genre = genre_counts.index[0]

            label_keywords = ", ".join(words).title() if words else top_genre
            self.topic_labels_map[t] = f"Topic {t}: {top_genre} ({label_keywords})"

    def get_topic_name(self, topic_id: int) -> str:
        return self.topic_labels_map.get(topic_id, f"Topic {topic_id}")

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds single user query string."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.model_name)
        vec = self.embedding_model.encode([text], normalize_embeddings=True)
        return vec[0]
