import streamlit as st
import pandas as pd
import numpy as np
from movie_recommender import MovieRecommenderSystem

# Set Streamlit page configuration
st.set_page_config(
    page_title="IMDb | Movie Recommender",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# IMDb-inspired sleek dark theme CSS
st.markdown(
    """
    <style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0F0F0F;
        color: #F5F5F5;
        font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Top Navbar */
    .imdb-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #121212;
        padding: 12px 24px;
        border-radius: 8px;
        margin-bottom: 24px;
        border-bottom: 1px solid #282828;
    }
    .imdb-logo {
        background-color: #F5C518;
        color: #000000;
        font-weight: 900;
        font-size: 1.5rem;
        padding: 4px 12px;
        border-radius: 6px;
        letter-spacing: -0.5px;
        display: inline-block;
        margin-right: 12px;
    }
    .imdb-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
        vertical-align: middle;
    }
    .imdb-tagline {
        color: #8E8E8E;
        font-size: 0.85rem;
    }

    /* Hero Spotlight Card */
    .hero-card {
        background: linear-gradient(145deg, #1A1A1A, #141414);
        border: 1px solid #2A2A2A;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    .hero-meta {
        color: #AAAAAA;
        font-size: 0.95rem;
        margin-bottom: 14px;
    }
    .rating-badge {
        display: inline-flex;
        align-items: center;
        background-color: rgba(245, 197, 24, 0.15);
        color: #F5C518;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 1rem;
        margin-right: 12px;
        border: 1px solid rgba(245, 197, 24, 0.3);
    }
    .genre-pill {
        display: inline-block;
        background-color: #262626;
        color: #D6D6D6;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #383838;
    }
    .topic-pill {
        display: inline-block;
        background-color: rgba(74, 144, 226, 0.15);
        color: #6BA4E8;
        border: 1px solid rgba(74, 144, 226, 0.3);
        padding: 3px 12px;
        border-radius: 14px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Movie Recommendation Card Grid */
    .rec-card {
        background-color: #1A1A1A;
        border: 1px solid #2A2A2A;
        border-radius: 10px;
        padding: 12px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-4px);
        border-color: #F5C518;
        box-shadow: 0 10px 20px rgba(245, 197, 24, 0.15);
    }
    .rec-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 10px;
        margin-bottom: 4px;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .rec-sub {
        color: #9E9E9E;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
    .match-pill {
        display: inline-block;
        background: linear-gradient(90deg, #10B981, #059669);
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .rec-reason {
        background-color: #121212;
        border-left: 3px solid #F5C518;
        padding: 6px 10px;
        font-size: 0.78rem;
        color: #CCCCCC;
        border-radius: 0 4px 4px 0;
        margin-top: 8px;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        font-size: 1.4rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 30px;
        margin-bottom: 18px;
        letter-spacing: -0.3px;
    }
    .section-header::before {
        content: "";
        display: inline-block;
        width: 4px;
        height: 22px;
        background-color: #F5C518;
        margin-right: 10px;
        border-radius: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading IMDb Movie Recommender Database...")
def get_engine():
    engine = MovieRecommenderSystem(csv_path="data/movies.csv", min_topic_size=15)
    engine.fit(force_retrain=False)
    return engine


engine = get_engine()
df = engine.df

# Session state initialization
if "selected_movie" not in st.session_state:
    st.session_state["selected_movie"] = "Inception"

# IMDb Top Bar
st.markdown(
    """
    <div class="imdb-navbar">
        <div>
            <span class="imdb-logo">IMDb</span>
            <span class="imdb-title">TopicRecommender</span>
            <span class="imdb-tagline"> | Powered by BERTopic Neural Embeddings</span>
        </div>
        <div style="color: #F5C518; font-weight: 700; font-size: 0.9rem;">
            9,737 Films Indexed
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation tabs like IMDb
nav_tab1, nav_tab2, nav_tab3, nav_tab4 = st.tabs(
    ["🔍 Search & 'More Like This'", "🍿 Recommended For You", "⚡ Mood / Vibe Search", "🏷️ Browse Topics"]
)


# =========================================================
# TAB 1: Search & "More Like This" (Core IMDb View)
# =========================================================
with nav_tab1:
    # Prominent Central Search Bar
    all_titles = sorted(df["Title"].dropna().unique().tolist())
    current_movie = st.session_state["selected_movie"]
    curr_idx = all_titles.index(current_movie) if current_movie in all_titles else 0

    col_search, col_btn = st.columns([5, 1])
    with col_search:
        search_selection = st.selectbox(
            "Search IMDb Movies:",
            options=all_titles,
            index=curr_idx,
            label_visibility="collapsed",
            placeholder="🔎 Search movies (e.g. Inception, The Dark Knight, Toy Story, Pulp Fiction...)",
        )
    with col_btn:
        sort_mode = st.selectbox(
            "Sort by",
            ["Top Match", "Highest Rated", "Newest"],
            label_visibility="collapsed",
        )

    # Optional Collapsed Settings (All sliders hidden by default)
    with st.expander("⚙️ Advanced Filter & Settings (Optional)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            rec_count = st.slider("Number of recommendations", 4, 16, 8, step=2)
        with c2:
            min_rating_filter = st.slider("Minimum IMDb Rating", 0.0, 9.0, 5.0, step=0.5)
        with c3:
            topic_boost_weight = st.slider("Topic affinity boost", 0.0, 0.4, 0.15, step=0.05)

    if not rec_count:
        rec_count = 8
    if not min_rating_filter:
        min_rating_filter = 5.0
    if not topic_boost_weight:
        topic_boost_weight = 0.15

    # Run recommendation
    res = engine.recommend_by_movie(
        search_selection,
        top_n=rec_count,
        topic_boost=topic_boost_weight,
        min_rating=min_rating_filter,
    )

    if "error" in res:
        st.error(res["error"])
    else:
        target = res["target_movie"]

        # ---------------- HERO MOVIE SPOTLIGHT ----------------
        with st.container():
            col_poster, col_details = st.columns([1, 3.2])
            with col_poster:
                poster_url = target.get("poster", "")
                if isinstance(poster_url, str) and poster_url.startswith("http"):
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(
                        """
                        <div style="background-color: #222; height: 320px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #666;">
                            🎞️ No Poster
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with col_details:
                st.markdown(f'<div class="hero-title">{target["title"]}</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="hero-meta">
                        <span class="rating-badge">★ {target['rating']} / 10</span>
                        <span>{target['year']}</span> • 
                        <span class="topic-pill">🏷️ {target['topic_name']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Genres pills
                genres_html = "".join([f'<span class="genre-pill">{g.strip()}</span>' for g in target["genre"].split(",")])
                st.markdown(f'<div style="margin-bottom: 12px;">{genres_html}</div>', unsafe_allow_html=True)

                if target.get("plot"):
                    st.markdown(f"<p style='color: #DDD; font-size: 1.02rem; line-height: 1.5;'>{target['plot']}</p>", unsafe_allow_html=True)

                meta_details = []
                if target.get("director"):
                    meta_details.append(f"<b>Director:</b> {target['director']}")
                if target.get("actors"):
                    meta_details.append(f"<b>Stars:</b> {target['actors']}")
                if meta_details:
                    st.markdown("<p style='color: #BBB; font-size: 0.9rem;'>" + " &nbsp;|&nbsp; ".join(meta_details) + "</p>", unsafe_allow_html=True)

        # ---------------- MORE LIKE THIS SECTION ----------------
        st.markdown('<div class="section-header">More Like This</div>', unsafe_allow_html=True)

        recs = res["recommendations"]

        # Sort recommendations if user requested
        if sort_mode == "Highest Rated":
            recs = sorted(recs, key=lambda x: x["rating"], reverse=True)
        elif sort_mode == "Newest":
            recs = sorted(recs, key=lambda x: x["year"], reverse=True)

        # Responsive Grid (4 columns)
        cols_per_row = 4
        for row_start in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            row_items = recs[row_start : row_start + cols_per_row]
            for c_idx, rec in enumerate(row_items):
                with cols[c_idx]:
                    with st.container():
                        # Poster
                        p_url = rec.get("poster", "")
                        if isinstance(p_url, str) and p_url.startswith("http"):
                            st.image(p_url, use_container_width=True)
                        else:
                            st.markdown(
                                """
                                <div style="background-color: #222; height: 260px; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #555;">
                                    🎞️ Poster
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        match_pct = int(min(max(rec["composite_score"], 0.0), 1.0) * 100)
                        st.markdown(
                            f"""
                            <div style="margin-top: 8px;">
                                <span class="match-pill">{match_pct}% Match</span>
                                <span style="color: #F5C518; font-weight: 700; font-size: 0.9rem;">★ {rec['rating']}</span>
                                <span style="color: #888; font-size: 0.85rem;">({rec['year']})</span>
                            </div>
                            <div class="rec-title">{rec['title']}</div>
                            <div class="rec-sub">{rec['genre']}</div>
                            <div class="rec-reason">💡 {rec['reasons'][0] if rec['reasons'] else 'High thematic match'}</div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Quick button to pivot and explore this movie
                        if st.button("Explore", key=f"pivot_{rec['title']}_{row_start}_{c_idx}", use_container_width=True):
                            st.session_state["selected_movie"] = rec["title"]
                            st.rerun()


# =========================================================
# TAB 2: Recommended For You (Personalized Profile)
# =========================================================
with nav_tab2:
    st.markdown('<div class="section-header">Personalized For You</div>', unsafe_allow_html=True)
    st.write("Tell us a few movies you like to get a personalized feed based on your aggregate taste vector.")

    col_u1, col_u2 = st.columns([3, 2])
    with col_u1:
        liked_picks = st.multiselect(
            "Add movies you loved:",
            options=all_titles,
            default=[m for m in ["The Dark Knight", "Inception", "Interstellar"] if m in all_titles][:2],
            placeholder="Type and select films...",
        )
    with col_u2:
        all_genres = sorted(list(set(df["Genre"].dropna().str.split(",").explode().str.strip().unique())))
        liked_genres = st.multiselect(
            "Favorite genres (optional boost):",
            options=all_genres,
            placeholder="e.g. Sci-Fi, Crime, Animation...",
        )

    if st.button("Generate My Personalized Feed", type="primary"):
        if not liked_picks and not liked_genres:
            st.warning("Please choose at least one movie or genre.")
        else:
            user_res = engine.recommend_for_user(
                liked_movies=liked_picks,
                preferred_genres=liked_genres,
                top_n=8,
                min_rating=5.5,
            )
            profile = user_res["taste_profile"]
            if profile.get("top_affinity_topics"):
                st.markdown(
                    f"<div style='background-color: #1A1A1A; border: 1px solid #333; padding: 10px 16px; border-radius: 8px; margin-bottom: 20px; color: #BBB;'>"
                    f"🎯 <b>Detected Taste Topics:</b> {', '.join(profile['top_affinity_topics'])}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            u_recs = user_res["recommendations"]
            cols_per_row = 4
            for row_start in range(0, len(u_recs), cols_per_row):
                cols = st.columns(cols_per_row)
                row_items = u_recs[row_start : row_start + cols_per_row]
                for c_idx, rec in enumerate(row_items):
                    with cols[c_idx]:
                        p_url = rec.get("poster", "")
                        if isinstance(p_url, str) and p_url.startswith("http"):
                            st.image(p_url, use_container_width=True)
                        match_pct = int(min(max(rec["match_score"], 0.0), 1.0) * 100)
                        st.markdown(
                            f"""
                            <div style="margin-top: 8px;">
                                <span class="match-pill">{match_pct}% Match</span>
                                <span style="color: #F5C518; font-weight: 700; font-size: 0.9rem;">★ {rec['rating']}</span>
                                <span style="color: #888; font-size: 0.85rem;">({rec['year']})</span>
                            </div>
                            <div class="rec-title">{rec['title']}</div>
                            <div class="rec-sub">{rec['genre']}</div>
                            <div class="rec-reason">🏷️ {rec['topic_name']}</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button("Explore", key=f"user_pivot_{rec['title']}_{row_start}_{c_idx}", use_container_width=True):
                            st.session_state["selected_movie"] = rec["title"]
                            st.rerun()


# =========================================================
# TAB 3: Mood & Vibe Search (Zero-Shot)
# =========================================================
with nav_tab3:
    st.markdown('<div class="section-header">Zero-Shot Mood & Vibe Search</div>', unsafe_allow_html=True)
    st.write("Search by describing a specific plot idea, atmosphere, or vibe in plain English.")

    popular_prompts = [
        "A dark mystery where a clever detective investigates in a rainy noir city",
        "Heartwarming high school coming of age comedy with quirky romance",
        "Mind-bending space sci-fi where astronauts question reality",
        "Terrifying psychological horror with an ancient haunted house",
        "High-stakes casino heist with sarcastic witty thieves",
    ]
    prompt_select = st.selectbox("Quick suggestions:", ["-- Choose an example or type below --"] + popular_prompts)
    default_text = prompt_select if prompt_select != "-- Choose an example or type below --" else "A tense psychological thriller with mind games and detectives"

    vibe_query = st.text_input("Describe what you feel like watching:", value=default_text)

    if st.button("Search By Vibe", type="primary"):
        if vibe_query.strip():
            v_res = engine.recommend_by_mood_or_query(vibe_query, top_n=8, min_rating=5.5)
            st.markdown(f"#### Top Matches for: *\"{vibe_query}\"*")
            v_recs = v_res["recommendations"]

            cols_per_row = 4
            for row_start in range(0, len(v_recs), cols_per_row):
                cols = st.columns(cols_per_row)
                row_items = v_recs[row_start : row_start + cols_per_row]
                for c_idx, rec in enumerate(row_items):
                    with cols[c_idx]:
                        p_url = rec.get("poster", "")
                        if isinstance(p_url, str) and p_url.startswith("http"):
                            st.image(p_url, use_container_width=True)
                        sem_pct = int(min(max(rec["semantic_score"], 0.0), 1.0) * 100)
                        st.markdown(
                            f"""
                            <div style="margin-top: 8px;">
                                <span class="match-pill">{sem_pct}% Semantic Vibe</span>
                                <span style="color: #F5C518; font-weight: 700; font-size: 0.9rem;">★ {rec['rating']}</span>
                                <span style="color: #888; font-size: 0.85rem;">({rec['year']})</span>
                            </div>
                            <div class="rec-title">{rec['title']}</div>
                            <div class="rec-sub">{rec['genre']}</div>
                            <div class="rec-reason">📖 {rec['plot'][:90]}...</div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button("Explore", key=f"mood_pivot_{rec['title']}_{row_start}_{c_idx}", use_container_width=True):
                            st.session_state["selected_movie"] = rec["title"]
                            st.rerun()


# =========================================================
# TAB 4: Browse Topics
# =========================================================
with nav_tab4:
    st.markdown('<div class="section-header">Curated BERTopic Categories</div>', unsafe_allow_html=True)
    st.write("Browse films grouped into semantic topic categories discovered by Class-based TF-IDF.")

    summary_df = engine.get_topic_summary()
    st.dataframe(
        summary_df[["Topic ID", "Topic Name", "Movies Count", "Dominant Genres", "Top Rated Movies"]],
        use_container_width=True,
        height=320,
    )

    t_choice = st.selectbox("Select Topic to inspect films:", summary_df["Topic ID"].tolist())
    topic_df = df[df["topic"] == t_choice].sort_values(by="weighted_rating", ascending=False).head(12)

    cols_per_row = 4
    for row_start in range(0, len(topic_df), cols_per_row):
        cols = st.columns(cols_per_row)
        row_items = topic_df.iloc[row_start : row_start + cols_per_row]
        for c_idx, (_, row) in enumerate(row_items.iterrows()):
            with cols[c_idx]:
                p_url = row.get("Poster", "")
                if isinstance(p_url, str) and p_url.startswith("http"):
                    st.image(p_url, use_container_width=True)
                st.markdown(
                    f"""
                    <div style="margin-top: 6px;">
                        <span style="color: #F5C518; font-weight: 700;">★ {row['clean_rating']}</span>
                        <span style="color: #888; font-size: 0.85rem;">({int(row['clean_year'])})</span>
                    </div>
                    <div class="rec-title">{row['Title']}</div>
                    <div class="rec-sub">{row['Genre']}</div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Explore", key=f"topic_pivot_{row['Title']}_{row_start}_{c_idx}", use_container_width=True):
                    st.session_state["selected_movie"] = row["Title"]
                    st.rerun()
