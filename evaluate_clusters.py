"""
Clustering Evaluation Script
Evaluates clustering quality using Silhouette Score, Davies-Bouldin Index, 
Calinski-Harabasz Index, and generates Elbow Method inertia values.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from movie_recommender.preprocessor import MovieDataPreprocessor
from movie_recommender.clusterer import MovieClusterer


def evaluate():
    print("=" * 60)
    print("CLUSTERING ALGORITHM EVALUATION")
    print("=" * 60)

    # 1. Load data and extract features
    preprocessor = MovieDataPreprocessor("data/movies.csv")
    df = preprocessor.load_data()
    print(f"Loaded {len(df)} movies from dataset.")

    clusterer = MovieClusterer(n_clusters=20)
    clusterer.fit_transform(df)
    X = clusterer.X_norm

    # 2. Compute metrics for current model (k=20)
    labels_20 = clusterer.cluster_labels
    sil = silhouette_score(X, labels_20, metric="cosine", sample_size=3000, random_state=42)
    db = davies_bouldin_score(X, labels_20)
    ch = calinski_harabasz_score(X, labels_20)

    print("\nMetrics for k=20:")
    print(f" - Silhouette Score (Cosine, sample=3000): {sil:.4f}")
    print(f" - Davies-Bouldin Index (lower is better): {db:.4f}")
    print(f" - Calinski-Harabasz Index (higher is better): {ch:.2f}")

    # 3. Elbow and Silhouette analysis across range of k
    print("\nEvaluating k across [5, 10, 15, 20, 25, 30]...")
    k_values = [5, 10, 15, 20, 25, 30]
    results = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        k_labels = km.fit_predict(X)
        k_sil = silhouette_score(X, k_labels, metric="cosine", sample_size=2000, random_state=42)
        k_db = davies_bouldin_score(X, k_labels)
        results.append({
            "k": k,
            "Inertia": round(km.inertia_, 2),
            "Silhouette": round(k_sil, 4),
            "Davies-Bouldin": round(k_db, 4),
        })
        print(f"  k={k:2d} | Inertia={km.inertia_:8.2f} | Silhouette={k_sil:.4f} | DB={k_db:.4f}")

    df_results = pd.DataFrame(results)
    print("\nComparison Table:")
    print(df_results.to_string(index=False))
    return df_results


if __name__ == "__main__":
    evaluate()
