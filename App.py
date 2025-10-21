import pandas as pd
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="NewsSphere", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Config ---
DATA_FILE_PATH = "./news.csv"
NUM_NEWS_TO_RATE = 5
RECOMMENDATION_COUNT = 10

CATEGORY_EMOJIS = {
    'BUSINESS': '📈', 'SPORTS': '⚽', 'WORLD': '🌎', 'US': '🇺🇸', 
    'HEALTH': '⚕️', 'POLITICS': '🏛️', 'ENTERTAINMENT': '🎬', 
    'TECHNOLOGY': '💻', 'SCIENCE': '🔬', 'TRAVEL': '✈️', 
    'LIFESTYLE': '🧘', 'CULTURE': '🎨', 'OPINION': '🗣️', 'OTHER': '📰',
    'KIDS': '👶🏻', 'WEATHER': '🌤️', 'FOODANDDRINK': '🍛🥛', 
    'FINANCE': '💰', 'MOVIES': '🎬'
}

def get_category_display(category):
    emoji = CATEGORY_EMOJIS.get(str(category).upper(), CATEGORY_EMOJIS['OTHER'])
    return f'<h4 style="color: #4B0082; font-weight: bold;">{emoji} {category}</h4>'

# --- CSS ---
st.markdown("""
<style>
.stApp > header { visibility: hidden; }
.article-card {
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    background-color: #fdfdfd;
}
.stButton>button { font-weight: bold; font-size: 1em; }
</style>
""", unsafe_allow_html=True)

# --- Data Functions ---
@st.cache_data
def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')
    except:
        df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', engine='python')
    df = df.rename(columns={
        df.columns[0]: "id", df.columns[1]: "category", df.columns[2]: "subcategory",
        df.columns[3]: "headline", df.columns[4]: "summary", df.columns[5]: "link",
    })
    df["headline"] = df["headline"].fillna('')
    df["summary"] = df["summary"].fillna('')
    df["category"] = df["category"].fillna('OTHER').astype(str).str.strip().str.upper()
    if "likes" not in df.columns:
        df["likes"] = 0
    return df

@st.cache_data
def fit_vectorizer_on_category(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    cat_matrix = vectorizer.fit_transform(df['category'].astype(str))
    return vectorizer, cat_matrix

def sample_unique_category_articles(df, n, exclude_ids=None, exclude_cats=None):
    if exclude_ids is None:
        exclude_ids = set()
    if exclude_cats is None:
        exclude_cats = set()

    selected = []
    used_categories = set()
    categories = [c for c in df['category'].dropna().unique() if c not in exclude_cats]
    random.shuffle(categories)
    for cat in categories:
        if cat in used_categories:
            continue
        pool = df[(df['category'] == cat) & (~df['id'].isin(exclude_ids))]
        if not pool.empty:
            selected.append(pool.sample(1).iloc[0].to_dict())
            used_categories.add(cat)
        if len(selected) == n:
            break
    if len(selected) < n:
        remaining = df[~df['id'].isin(exclude_ids.union({r['id'] for r in selected}))] 
        for _, row in remaining.sample(min(n-len(selected), len(remaining))).iterrows():
            selected.append(row.to_dict())
            if len(selected) == n:
                break
    return selected

def calculate_recommendations(df, cat_matrix, session_likes, rated_ids):
    if not session_likes:
        return None

    liked_ids = list(session_likes.keys())
    liked_cats = df[df['id'].isin(liked_ids)]['category'].str.strip().str.upper().unique().tolist()
    liked_cats = [cat for cat in liked_cats if cat != "KIDS"]

    candidate_df = df[df['category'].str.strip().str.upper().isin(liked_cats) & ~df['id'].isin(rated_ids)].copy()

    if candidate_df.empty:
        candidate_df = df[df['category'].str.strip().str.upper().isin(liked_cats)].copy()

    candidate_df = candidate_df.sample(frac=1).reset_index(drop=True)

    if not candidate_df.empty:
        max_len = max(candidate_df['summary'].apply(lambda x: len(x.split())))
        for idx, row in candidate_df.iterrows():
            words = row['summary'].split()
            if len(words) < max_len:
                extra_needed = max_len - len(words)
                extras = " ".join(words[:extra_needed])
                candidate_df.at[idx, 'summary'] = row['summary'] + " " + extras

    return candidate_df.head(RECOMMENDATION_COUNT)[["id","category","headline","summary"]]

# --- Session State ---
def init_session_state():
    if 'rated_count' not in st.session_state: st.session_state.rated_count = 0
    if 'session_likes' not in st.session_state: st.session_state.session_likes = {}
    if 'rated_ids' not in st.session_state: st.session_state.rated_ids = set()
    if 'rating_queue' not in st.session_state: st.session_state.rating_queue = []
    if 'current_index' not in st.session_state: st.session_state.current_index = 0
    if 'disliked_cats' not in st.session_state: st.session_state.disliked_cats = set()

def handle_like():
    if st.session_state.current_index < len(st.session_state.rating_queue):
        cur = st.session_state.rating_queue[st.session_state.current_index]
        st.session_state.session_likes[cur['id']] = 1
        st.session_state.rated_ids.add(cur['id'])
        st.session_state.rated_count += 1
        st.session_state.current_index += 1
    else:
        st.warning("No more articles to rate.")
        st.session_state.current_index = len(st.session_state.rating_queue)

def handle_dislike():
    if st.session_state.current_index < len(st.session_state.rating_queue):
        cur = st.session_state.rating_queue[st.session_state.current_index]
        st.session_state.rated_ids.add(cur['id'])
        st.session_state.disliked_cats.add(cur['category'])
        st.session_state.rated_count += 1
        st.session_state.current_index += 1
    else:
        st.warning("No more articles to rate.")
        st.session_state.current_index = len(st.session_state.rating_queue)

def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# --- Main App ---
def app():
    st.markdown('<h1 style="text-align:center; color:#4B0082;">🌍 NewsSphere</h1>', unsafe_allow_html=True)
    st.markdown("🌊 Ride the wave of news and stories you’ll love—dive in now!")
    st.markdown("---")

    df = load_and_process_data(DATA_FILE_PATH)
    _, cat_matrix = fit_vectorizer_on_category(df)

    init_session_state()

    if not st.session_state.rating_queue:
        st.session_state.rating_queue = sample_unique_category_articles(df, n=NUM_NEWS_TO_RATE)
        st.session_state.current_index = 0
        st.session_state.rated_count = 0
        st.session_state.session_likes = {}
        st.session_state.rated_ids = set()
        st.session_state.disliked_cats = set()

    # ---------- Rating Phase ----------
    if st.session_state.rated_count < NUM_NEWS_TO_RATE:
        progress_percent = int((st.session_state.rated_count / NUM_NEWS_TO_RATE) * 100)
        st.progress(progress_percent)

        if st.session_state.current_index >= len(st.session_state.rating_queue):
            st.warning("No more articles to rate. Click below to continue.")
            if st.button("➡️ Continue"):
                st.session_state.rated_count = NUM_NEWS_TO_RATE
                st.rerun()
        else:
            cur = st.session_state.rating_queue[st.session_state.current_index]
            st.markdown(get_category_display(cur['category']), unsafe_allow_html=True)
            st.markdown(f"## {cur['headline']}")
            st.markdown(f"{cur['summary']}")
            col1, col2, col3 = st.columns([1,4,1])
            with col1: st.button("👍 Like", key=f"like_{cur['id']}", on_click=handle_like, use_container_width=True)
            with col3: st.button("👎 Dislike", key=f"dislike_{cur['id']}", on_click=handle_dislike, use_container_width=True)

    # ---------- Recommendation Phase ----------
    elif st.session_state.rated_count >= NUM_NEWS_TO_RATE:
        st.markdown("---")

        if not st.session_state.session_likes:
            st.warning("You disliked all articles 😅 Let's try something new from other categories!")
            st.session_state.rating_queue = sample_unique_category_articles(
                df, n=NUM_NEWS_TO_RATE, exclude_cats=st.session_state.disliked_cats
            )
            st.session_state.current_index = 0
            st.session_state.rated_count = 0
            st.session_state.session_likes = {}
            st.session_state.rated_ids = set()
            st.session_state.disliked_cats = set()
            st.rerun()

        st.markdown('<h2 style="text-align:center; color:#4B0082;">✨📰 Here are some exciting articles just for you! 🚀📚</h2>', unsafe_allow_html=True)
        recommended_df = calculate_recommendations(df, cat_matrix, st.session_state.session_likes, st.session_state.rated_ids)

        if recommended_df is None or recommended_df.empty:
            st.warning("No recommendations available from your liked categories.")
            if st.button("🔄 Restart"):
                clear_session_state()
                st.rerun()
        else:
            for idx, row in recommended_df.iterrows():
                st.markdown(f"<div class='article-card'>", unsafe_allow_html=True)
                st.markdown(get_category_display(row['category']), unsafe_allow_html=True)
                st.markdown(f"### {row['headline']}")
                st.markdown(f"{row['summary']}")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")

if __name__ == "__main__":
    app()
