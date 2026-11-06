import time
from movie_recommender import MovieRecommenderSystem


def run():
    print("=" * 70)
    print("BERTopic Movie Recommender System - Training & Evaluation")
    print("=" * 70)

    start = time.time()
    engine = MovieRecommenderSystem(csv_path="data/movies.csv", min_topic_size=15)
    engine.fit(force_retrain=False)
    elapsed = time.time() - start
    print(f"\n[OK] Engine ready in {elapsed:.1f}s! Total movies: {len(engine.df)}")

    # 1. Topic Summary
    print("\n" + "=" * 70)
    print("DISCOVERED TOPICS SAMPLE (Top 8):")
    print("=" * 70)
    summary = engine.get_topic_summary()
    print(summary.head(8).to_string(index=False))

    # 2. Movie-to-Movie Recommendation: Inception
    print("\n" + "=" * 70)
    print("TEST 1: Movie-to-Movie Recommendation for 'Inception'")
    print("=" * 70)
    res = engine.recommend_by_movie("Inception", top_n=5)
    target = res["target_movie"]
    print(f"Target: {target['title']} ({target['year']}) | Topic: {target['topic_name']}")
    for r in res["recommendations"]:
        print(f"#{r['rank']} {r['title']} ({r['year']}) | IMDb: {r['rating']}/10 | Sim: {r['similarity_score']} | Match: {r['composite_score']}")
        print(f"   Topic: {r['topic_name']}")
        print(f"   Why: {', '.join(r['reasons'])}")

    # 3. User Profile Recommendation
    print("\n" + "=" * 70)
    print("TEST 2: Personalized User Profile -> Likes ['Toy Story', 'Finding Nemo']")
    print("=" * 70)
    user_res = engine.recommend_for_user(liked_movies=["Toy Story", "Finding Nemo"], top_n=5)
    print(f"Taste Profile: {user_res['taste_profile']}")
    for r in user_res["recommendations"]:
        print(f"#{r['rank']} {r['title']} ({r['year']}) | Score: {r['match_score']} | Topic: {r['topic_name']} | Genre: {r['genre']}")

    # 4. Zero-Shot Natural Language Mood Recommendation
    print("\n" + "=" * 70)
    print("TEST 3: Natural Language Mood Search -> 'creepy haunted house and demonic paranormal investigation'")
    print("=" * 70)
    mood_res = engine.recommend_by_mood_or_query("creepy haunted house and demonic paranormal investigation", top_n=5)
    for r in mood_res["recommendations"]:
        print(f"#{r['rank']} {r['title']} ({r['year']}) | Sem: {r['semantic_score']} | Score: {r['match_score']} | Genre: {r['genre']}")


if __name__ == "__main__":
    run()
