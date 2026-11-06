import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from movie_recommender import MovieRecommenderSystem

st.set_page_config(
    page_title="Movie Recommender & BERTopic Explorer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    .movie-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .movie-card:hover {
        border-color: #FF4B4B;
        transform: translateY(-2px);
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #FF4B4B;
        color: white;
        margin-right: 6px;
    }
    .badge-topic {
        background-color: #4A90E2;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading BERTopic Movie Recommender & Embeddings...")
def get_engine():
    engine = MovieRecommenderSystem(csv_path="data/movies.csv", min_topic_size=15)
    engine.fit(force_retrain=False)
    return engine


engine = get_engine()
df = engine.df

# Header
st.markdown('<h1 class="main-title">🎬 BERTopic Movie Recommender</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Semantic Topic Modeling & Neural Embedding Recommendations on 9,700+ Films</p>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.image(
    "https://raw.githubusercontent.com/MaartenGr/BERTopic/master/docs/images/logo.png",
    width=180,
)
st.sidebar.markdown("### Navigation")
mode = st.sidebar.radio(
    "Choose Mode:",
    [
        "🎯 Movie-to-Movie Recommender",
        "👤 Personalized Profile (For People)",
        "✨ Natural Language Mood Search",
        "🌌 Interactive Topic Galaxy (2D Map)",
        "📊 Discovered Topics Explorer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Architecture")
num_topics = len([t for t in df["topic"].unique() if t != -1])
st.sidebar.metric("Total Movies", f"{len(df):,}")
st.sidebar.metric("Discovered Topics", f"{num_topics}")
st.sidebar.metric("Embedding Dim", "384-D (MiniLM)")
st.sidebar.markdown("---")
st.sidebar.caption("Pipeline: Sentence-Transformers → UMAP → HDBSCAN → c-TF-IDF")

# Helper to render movie card
def render_movie_card(m, is_target=False):
    col_img, col_info = st.columns([1, 3])
    with col_img:
        poster = m.get("poster", "")
        if isinstance(poster, str) and poster.startswith("http"):
            st.image(poster, use_container_width=True)
        else:
            st.markdown("🖼️ *No Poster Available*")
    with col_info:
        badge_label = "TARGET FILM" if is_target else f"#{m.get('rank', '')} MATCH"
        st.markdown(
            f"<span class='badge'>{badge_label}</span> <span class='badge badge-topic'>{m.get('topic_name', 'Topic')}</span>",
            unsafe_allow_html=True,
        )
        st.subheader(f"{m.get('title')} ({m.get('year')})")
        st.markdown(f"⭐ **IMDb:** {m.get('rating')} / 10 | **Genres:** {m.get('genre')}")
        if m.get("director"):
            st.markdown(f"🎬 **Director:** {m.get('director')}")
        if m.get("actors"):
            st.markdown(f"👥 **Cast:** {m.get('actors')}")
        if m.get("composite_score"):
            st.progress(min(max(float(m["composite_score"]), 0.0), 1.0), text=f"Match Score: {int(m['composite_score']*100)}%")
        if m.get("reasons"):
            st.markdown("💡 **Why recommended:**")
            for r in m["reasons"]:
                st.markdown(f"- {r}")
        if m.get("plot"):
            st.caption(f"📖 **Synopsis:** {m['plot']}")


# ==========================================
# MODE 1: Movie to Movie Recommender
# ==========================================
if mode == "🎯 Movie-to-Movie Recommender":
    st.markdown("### 🎯 Find Movies Similar to a Film You Love")
    st.write("Pick any movie in the collection to discover semantically and thematically related recommendations.")

    movie_titles = sorted(df["Title"].dropna().unique().tolist())
    default_idx = movie_titles.index("Inception") if "Inception" in movie_titles else 0
    selected_movie = st.selectbox("Search or select a movie:", movie_titles, index=default_idx)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        top_n = st.slider("Number of recommendations", 3, 15, 6)
    with col_s2:
        topic_boost = st.slider("Topic affinity boost", 0.0, 0.4, 0.15, step=0.05)
    with col_s3:
        min_rating = st.slider("Minimum IMDb rating", 0.0, 9.0, 5.5, step=0.5)

    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        res = engine.recommend_by_movie(
            selected_movie,
            top_n=top_n,
            topic_boost=topic_boost,
            min_rating=min_rating,
        )
        if "error" in res:
            st.error(res["error"])
        else:
            st.markdown("#### Selected Movie")
            render_movie_card(res["target_movie"], is_target=True)
            st.markdown("---")
            st.markdown(f"#### Top {top_n} Recommendations")
            for rec in res["recommendations"]:
                with st.container():
                    render_movie_card(rec)
                    st.divider()


# ==========================================
# MODE 2: Personalized Profile for People
# ==========================================
elif mode == "👤 Personalized Profile (For People)":
    st.markdown("### 👤 Personalized Recommendations for Your Taste Profile")
    st.write("Select multiple films you enjoy and optional genres to compute your aggregate taste vector.")

    movie_titles = sorted(df["Title"].dropna().unique().tolist())
    defaults = [m for m in ["The Dark Knight", "Inception", "Interstellar", "The Matrix"] if m in movie_titles]
    liked_selected = st.multiselect(
        "Select movies you like / watched:",
        options=movie_titles,
        default=defaults[:3],
    )

    all_genres = sorted(
        list(
            set(
                df["Genre"]
                .dropna()
                .str.split(",")
                .explode()
                .str.strip()
                .unique()
            )
        )
    )
    pref_genres = st.multiselect("Favorite genres (optional boost):", options=all_genres)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        top_n_user = st.slider("Recommendations count", 4, 16, 8)
    with col_p2:
        min_user_rating = st.slider("Minimum IMDb rating", 0.0, 8.5, 6.5, step=0.5)

    if st.button("Build My Recommendations", type="primary", use_container_width=True):
        if not liked_selected and not pref_genres:
            st.warning("Please choose at least one movie or genre.")
        else:
            user_res = engine.recommend_for_user(
                liked_movies=liked_selected,
                preferred_genres=pref_genres,
                top_n=top_n_user,
                min_rating=min_user_rating,
            )
            profile = user_res["taste_profile"]
            st.success(f"Discovered Taste Clusters: {', '.join(profile['top_affinity_topics']) if profile['top_affinity_topics'] else 'General Semantic Centroid'}")
            st.markdown("---")
            for rec in user_res["recommendations"]:
                with st.container():
                    render_movie_card(rec)
                    st.divider()


# ==========================================
# MODE 3: Natural Language Mood Search
# ==========================================
elif mode == "✨ Natural Language Mood Search":
    st.markdown("### ✨ Zero-Shot Mood & Vibe Recommendation")
    st.write("Type what you feel like watching in plain English. The Sentence Transformer projects your mood into the BERTopic semantic embedding space to find matching films!")

    sample_prompts = [
        "A dark mystery where a clever detective investigates in a rainy noir city",
        "Heartwarming coming of age comedy about awkward teenagers finding love",
        "Mind-bending space sci-fi where scientists question what is reality",
        "Terrifying psychological horror with an ancient cursed mansion",
        "High-stakes casino heist with sarcastic witty thieves",
    ]
    prompt_choice = st.selectbox("Or choose an example prompt:", [""] + sample_prompts)
    user_query = st.text_input("Enter your movie mood or prompt:", value=prompt_choice if prompt_choice else "Mind-bending space sci-fi with emotional stakes")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mood_n = st.slider("Recommendations count", 4, 12, 6, key="mood_n")
    with col_m2:
        mood_min_rating = st.slider("Minimum IMDb rating", 0.0, 8.5, 6.0, step=0.5, key="mood_min_r")

    if st.button("Find Movies Matching Mood", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a mood description.")
        else:
            mood_res = engine.recommend_by_mood_or_query(user_query, top_n=mood_n, min_rating=mood_min_rating)
            st.markdown(f"#### Top Matches for: *\"{user_query}\"*")
            st.markdown("---")
            for rec in mood_res["recommendations"]:
                with st.container():
                    render_movie_card(rec)
                    st.divider()


# ==========================================
# MODE 4: Interactive Topic Galaxy (2D Map)
# ==========================================
elif mode == "🌌 Interactive Topic Galaxy (2D Map)":
    st.markdown("### 🌌 2D Semantic Topic Galaxy")
    st.write("Every point is a movie positioned by UMAP based on its dense BERTopic embedding. Movies clustered together share plot themes, tone, and personnel.")

    # Filter sample for responsiveness
    sample_df = df.copy()
    if len(sample_df) > 3000:
        sample_df = sample_df.sample(3000, random_state=42)

    sample_df["topic_label"] = sample_df["topic"].apply(lambda t: engine.topic_model.get_topic_name(t))

    fig = px.scatter(
        sample_df,
        x="coord_x",
        y="coord_y",
        color="topic_label",
        hover_name="Title",
        hover_data={"Year": True, "clean_rating": True, "Genre": True, "coord_x": False, "coord_y": False, "topic_label": True},
        title="BERTopic Movie Embeddings (UMAP 2D Projection)",
        opacity=0.75,
        height=700,
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# MODE 5: Discovered Topics Explorer
# ==========================================
elif mode == "📊 Discovered Topics Explorer":
    st.markdown("### 📊 Discovered Semantic Topics")
    st.write("Summary table of all topics extracted by Class-based TF-IDF (c-TF-IDF).")

    summary_df = engine.get_topic_summary()
    st.dataframe(summary_df, use_container_width=True, height=450)

    selected_t = st.selectbox("Inspect movies in topic:", summary_df["Topic ID"].tolist())
    topic_movies = df[df["topic"] == selected_t].sort_values(by="weighted_rating", ascending=False)
    st.write(f"Showing top rated films in **{engine.topic_model.get_topic_name(selected_t)}** ({len(topic_movies)} total movies):")

    st.dataframe(
        topic_movies[["Title", "clean_year", "clean_rating", "Genre", "Director", "Actors"]].rename(
            columns={"clean_year": "Year", "clean_rating": "IMDb Rating"}
        ).head(20),
        use_container_width=True,
    )
