import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from collections import Counter


class MovieClusterer:
    """Extracts features, runs KMeans clustering, and profiles movie clusters."""

    def __init__(self, n_clusters: int = 20, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=8000,
            min_df=2,
            ngram_range=(1, 2),
        )
        self.svd = TruncatedSVD(n_components=60, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.pca_2d = PCA(n_components=2, random_state=random_state)

        self.X_tfidf = None
        self.X_svd = None
        self.X_norm = None
        self.X_2d = None
        self.cluster_labels = None
        self.cluster_profiles = {}

    def fit_transform(self, df: pd.DataFrame):
        # 1. TF-IDF vectorization
        self.X_tfidf = self.tfidf.fit_transform(df["soup"])

        # 2. SVD reduction + L2 normalization (spherical space)
        self.X_svd = self.svd.fit_transform(self.X_tfidf)
        self.X_norm = normalize(self.X_svd, norm="l2")

        # 3. K-Means clustering
        self.cluster_labels = self.kmeans.fit_predict(self.X_norm)
        df["cluster"] = self.cluster_labels

        # 4. 2D projection for interactive visualization
        self.X_2d = self.pca_2d.fit_transform(self.X_svd)
        df["coord_x"] = self.X_2d[:, 0]
        df["coord_y"] = self.X_2d[:, 1]

        # 5. Profile each cluster
        self._build_cluster_profiles(df)
        return self.cluster_labels

    def _build_cluster_profiles(self, df: pd.DataFrame):
        feature_names = np.array(self.tfidf.get_feature_names_out())

        for c in range(self.n_clusters):
            c_mask = df["cluster"] == c
            c_df = df[c_mask]
            count = len(c_df)

            if count == 0:
                continue

            # Top genres
            all_genres = []
            for g_str in c_df["Genre"].dropna():
                all_genres.extend([g.strip() for g in g_str.split(",") if g.strip()])
            top_genres = [g for g, _ in Counter(all_genres).most_common(3)]

            # Top TF-IDF keywords in this cluster
            cluster_tfidf_mean = self.X_tfidf[c_mask.values].mean(axis=0)
            top_keyword_indices = np.argsort(np.asarray(cluster_tfidf_mean).flatten())[::-1][:8]
            top_keywords = [
                feature_names[idx].replace("_", " ").title()
                for idx in top_keyword_indices
                if len(feature_names[idx]) > 2
            ]

            # Top rated representative movies
            top_movies = (
                c_df.sort_values(by="weighted_rating", ascending=False)["Title"]
                .head(4)
                .tolist()
            )

            # Auto-generate a readable theme title
            theme = " / ".join(top_genres[:2]) if top_genres else "General"
            if top_keywords:
                theme += f" ({', '.join(top_keywords[:2])})"

            self.cluster_profiles[c] = {
                "cluster_id": c,
                "name": f"Cluster {c}: {theme}",
                "size": count,
                "top_genres": top_genres,
                "top_keywords": top_keywords[:6],
                "flagship_movies": top_movies,
            }

    def get_cluster_name(self, cluster_id: int) -> str:
        if cluster_id in self.cluster_profiles:
            return self.cluster_profiles[cluster_id]["name"]
        return f"Cluster {cluster_id}"
