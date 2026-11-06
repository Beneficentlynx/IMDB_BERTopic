"""
Interactive Command Line Interface for BERTopic Movie Recommender
"""

import sys
from movie_recommender import MovieRecommenderSystem


def print_banner():
    print(r"""
  ____  ______ _____ _______            _      
 |  _ \|  ____|  __ \__   __|          (_)     
 | |_) | |__  | |__) | | | ___  _ __  _  ___   
 |  _ <|  __| |  _  /  | |/ _ \| '_ \| |/ __|  
 | |_) | |____| | \ \  | | (_) | |_) | | (__   
 |____/|______|_|  \_\ |_|\___/| .__/|_|\___|  
   MOVIE RECOMMENDER & TOPICS  | |             
                               |_|             
    """)
    print("      Neural Topic Modeling (SentenceTransformers + UMAP + HDBSCAN + c-TF-IDF)")
    print("=" * 80)


def recommend_movie_interactive(engine: MovieRecommenderSystem):
    query = input("\nEnter a movie title you like (e.g. Inception, Toy Story, Pulp Fiction): ").strip()
    if not query:
        return

    matches = engine.search_movies(query, limit=5)
    if matches.empty:
        print(f"No movie found matching '{query}'. Try another search!")
        return

    selected_title = matches.iloc[0]["Title"]
    if len(matches) > 1:
        print(f"\nTop search matches for '{query}':")
        for i, (_, row) in enumerate(matches.iterrows(), 1):
            print(f"  [{i}] {row['Title']} ({row['Year']}) - {row['Genre']}")
        choice = input(f"Select movie [1-{len(matches)}] (default 1): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(matches):
            selected_title = matches.iloc[int(choice) - 1]["Title"]

    top_n = input("Number of recommendations (default 5): ").strip()
    top_n = int(top_n) if top_n.isdigit() and int(top_n) > 0 else 5

    res = engine.recommend_by_movie(selected_title, top_n=top_n)
    if "error" in res:
        print(f"Error: {res['error']}")
        return

    target = res["target_movie"]
    print("\n" + "=" * 80)
    print(f" TARGET: {target['title']} ({target['year']}) | IMDb: {target['rating']}/10")
    print(f" GENRE:  {target['genre']}")
    print(f" TOPIC:  {target['topic_name']}")
    print(f" PLOT:   {target['plot'][:120]}...")
    print("=" * 80)
    print(" RECOMMENDATIONS FOR YOU:")
    print("-" * 80)

    for r in res["recommendations"]:
        print(f"#{r['rank']} {r['title']} ({r['year']}) | IMDb: {r['rating']}/10 | Match: {r['composite_score']} (Sim: {r['similarity_score']})")
        print(f"   Topic:    {r['topic_name']}")
        print(f"   Genres:   {r['genre']}")
        print(f"   Director: {r['director']} | Cast: {r['actors']}")
        print(f"   Why:      {', '.join(r['reasons'])}")
        print()


def recommend_person_interactive(engine: MovieRecommenderSystem):
    print("\nBuild a custom taste profile for a person.")
    print("Enter 2 to 5 movies you love, separated by commas:")
    print("Example: The Dark Knight, Inception, Interstellar, The Matrix")
    raw = input("> ").strip()
    if not raw:
        return

    liked = [m.strip() for m in raw.split(",") if m.strip()]
    print("\nAny genres you especially love? (Optional, e.g. Sci-Fi, Crime, Animation):")
    genre_raw = input("> ").strip()
    genres = [g.strip() for g in genre_raw.split(",") if g.strip()] if genre_raw else None

    res = engine.recommend_for_user(liked_movies=liked, preferred_genres=genres, top_n=6)
    if "error" in res:
        print(f"Error: {res['error']}")
        return

    profile = res["taste_profile"]
    print("\n" + "=" * 80)
    print(f" LIKED MOVIES:    {', '.join(profile['liked_movies'])}")
    print(f" TASTE CLUSTERS:  {', '.join(profile['top_affinity_topics']) if profile['top_affinity_topics'] else 'General'}")
    print("=" * 80)
    print(" PERSONALIZED RECOMMENDATIONS:")
    print("-" * 80)

    for r in res["recommendations"]:
        print(f"#{r['rank']} {r['title']} ({r['year']}) | IMDb: {r['rating']}/10 | Match: {r['match_score']}")
        print(f"   Topic:    {r['topic_name']}")
        print(f"   Genres:   {r['genre']}")
        print(f"   Director: {r['director']}")
        print(f"   Plot:     {r['plot'][:100]}...")
        print()


def recommend_mood_interactive(engine: MovieRecommenderSystem):
    print("\nDescribe in plain English the kind of movie or mood you want:")
    print("Example: A tense psychological thriller with mind games and detectives")
    query = input("> ").strip()
    if not query:
        return

    res = engine.recommend_by_mood_or_query(query, top_n=6)
    print("\n" + "=" * 80)
    print(f" MOOD QUERY: \"{query}\"")
    print("=" * 80)
    print(" ZERO-SHOT NEURAL MATCHES:")
    print("-" * 80)

    for r in res["recommendations"]:
        print(f"#{r['rank']} {r['title']} ({r['year']}) | IMDb: {r['rating']}/10 | Semantic: {r['semantic_score']}")
        print(f"   Topic:    {r['topic_name']}")
        print(f"   Genres:   {r['genre']}")
        print(f"   Director: {r['director']}")
        print(f"   Plot:     {r['plot'][:100]}...")
        print()


def explore_topics_interactive(engine: MovieRecommenderSystem):
    print("\n" + "=" * 80)
    print(" DISCOVERED SEMANTIC TOPICS (c-TF-IDF)")
    print("=" * 80)
    summary = engine.get_topic_summary()
    for _, row in summary.iterrows():
        print(f"[{row['Topic ID']:2d}] {row['Topic Name']}")
        print(f"     Movies: {row['Movies Count']} | Genres: {row['Dominant Genres']}")
        print(f"     Top Rated: {row['Top Rated Movies']}")
        print("-" * 80)

    t_id = input("\nEnter Topic ID to see top 10 movies in that topic (or Enter to return): ").strip()
    if t_id.isdigit():
        tid = int(t_id)
        t_movies = (
            engine.df[engine.df["topic"] == tid]
            .sort_values(by="clean_rating", ascending=False)
            .head(10)
        )
        print(f"\nTop 10 rated movies in Topic {tid}:")
        for i, (_, m) in enumerate(t_movies.iterrows(), 1):
            print(f"  {i:2d}. {m['Title']} ({int(m['clean_year'])}) - IMDb: {m['clean_rating']}/10 | {m['Genre']}")


def main():
    print_banner()
    print("Initializing BERTopic Recommender System...")
    engine = MovieRecommenderSystem(csv_path="data/movies.csv", min_topic_size=15)
    engine.fit(force_retrain=False)
    print("[OK] Model ready!\n")

    while True:
        print("\nChoose an option:")
        print("  [1] Recommend movies based on a movie you like")
        print("  [2] Recommend movies for a person's taste profile (multi-movie)")
        print("  [3] Zero-shot natural language mood / vibe search")
        print("  [4] Explore discovered BERTopics")
        print("  [5] Search movie database")
        print("  [6] Exit")
        choice = input("\nEnter choice [1-6]: ").strip()

        if choice == "1":
            recommend_movie_interactive(engine)
        elif choice == "2":
            recommend_person_interactive(engine)
        elif choice == "3":
            recommend_mood_interactive(engine)
        elif choice == "4":
            explore_topics_interactive(engine)
        elif choice == "5":
            q = input("\nEnter title search term: ").strip()
            if q:
                res = engine.search_movies(q, limit=10)
                print(res.to_string(index=False))
        elif choice == "6" or choice.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please enter 1 to 6.")


if __name__ == "__main__":
    main()
