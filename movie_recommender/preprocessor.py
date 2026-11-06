import re
import pandas as pd
import numpy as np


class MovieDataPreprocessor:
    """Preprocesses raw movie dataset for topic modeling and recommendations."""

    def __init__(self, csv_path: str = "data/movies.csv"):
        self.csv_path = csv_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception:
            self.df = pd.read_csv(r"c:\Users\lucas\Downloads\movies.csv")

        # Deduplicate on Title and Year
        self.df = self.df.drop_duplicates(subset=["Title", "Year"]).reset_index(drop=True)
        self._clean_fields()
        return self.df

    def _clean_fields(self):
        # 1. Title
        self.df["Title"] = self.df["Title"].fillna("Unknown Title").astype(str)

        # 2. Year (extract 4 digits)
        self.df["clean_year"] = (
            self.df["Year"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
        )
        self.df["clean_year"] = self.df["clean_year"].fillna(self.df["clean_year"].median())

        # 3. IMDb Rating
        self.df["clean_rating"] = pd.to_numeric(self.df["imdbRating"], errors="coerce")
        median_rating = self.df["clean_rating"].median()
        self.df["clean_rating"] = self.df["clean_rating"].fillna(median_rating)

        # 4. IMDb Votes
        def _parse_votes(v):
            if pd.isna(v) or v == "N/A":
                return 0.0
            try:
                return float(str(v).replace(",", ""))
            except Exception:
                return 0.0

        self.df["clean_votes"] = self.df["imdbVotes"].apply(_parse_votes)

        # 5. Bayesian Weighted Rating
        C = self.df["clean_rating"].mean()
        m = 500.0
        v = self.df["clean_votes"]
        R = self.df["clean_rating"]
        self.df["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C
        min_wr = self.df["weighted_rating"].min()
        max_wr = self.df["weighted_rating"].max()
        self.df["quality_score"] = (self.df["weighted_rating"] - min_wr) / (max_wr - min_wr + 1e-8)

        # 6. Fill missing metadata
        self.df["Genre"] = self.df["Genre"].fillna("Unknown")
        self.df["Plot"] = self.df["Plot"].fillna("")
        self.df["Director"] = self.df["Director"].fillna("")
        self.df["Actors"] = self.df["Actors"].fillna("")
        self.df["Country"] = self.df["Country"].fillna("")
        self.df["Poster"] = self.df["Poster"].fillna("")

        # 7. Rich document formulation for BERTopic
        self.df["bertopic_doc"] = self._build_bertopic_documents()

    def _build_bertopic_documents(self) -> pd.Series:
        docs = []
        for _, row in self.df.iterrows():
            title = row["Title"]
            year = int(row["clean_year"])
            genre = row["Genre"]
            director = row["Director"]
            actors = row["Actors"]
            plot = row["Plot"]

            parts = [f"{title} ({year})."]
            if genre and genre != "Unknown":
                parts.append(f"Genres: {genre}.")
            if director and director != "N/A":
                parts.append(f"Directed by {director}.")
            if actors and actors != "N/A":
                parts.append(f"Starring: {actors}.")
            if plot and plot != "N/A":
                parts.append(f"Plot: {plot}")

            docs.append(" ".join(parts))
        return pd.Series(docs, index=self.df.index)
