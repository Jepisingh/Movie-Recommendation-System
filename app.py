import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

@st.cache_data
def load_and_process():
    # 1. Load CSV files
    movies = pd.read_csv("tmdb_5000_movies.csv.zip")
    credits = pd.read_csv("tmdb_5000_credits.csv.zip")

    # 2. Merge on title
    movies = movies.merge(credits, on="title")

    # 3. Fill NaN so ast.literal_eval and string ops don't break
    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("[]")
    movies["keywords"] = movies["keywords"].fillna("[]")
    movies["cast"] = movies["cast"].fillna("[]")
    movies["crew"] = movies["crew"].fillna("[]")

    # 4. Helper to convert JSON-like string columns (genres, keywords etc.)
    def convert(obj):
        try:
            L = []
            data = ast.literal_eval(obj)
            for i in data:
                if isinstance(i, dict) and "name" in i:
                    L.append(i["name"])
            return " ".join(L)
        except Exception:
            return ""

    # 5. Apply conversions
    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)

    def get_cast_top3(obj):
        try:
            data = ast.literal_eval(obj)
            names = [i["name"] for i in data[:3] if isinstance(i, dict) and "name" in i]
            return " ".join(names)
        except Exception:
            return ""

    movies["cast"] = movies["cast"].apply(get_cast_top3)

    def get_director(obj):
        try:
            data = ast.literal_eval(obj)
            for i in data:
                if isinstance(i, dict) and i.get("job") == "Director":
                    return i.get("name", "")
            return ""
        except Exception:
            return ""

    movies["crew"] = movies["crew"].apply(get_director)

    # 6. Create tags (make sure everything is string & no NaN)
    for col in ["overview", "genres", "keywords", "cast", "crew"]:
        movies[col] = movies[col].fillna("").astype(str)

    movies["tags"] = (
        movies["overview"]
        + " "
        + movies["genres"]
        + " "
        + movies["keywords"]
        + " "
        + movies["cast"]
        + " "
        + movies["crew"]
    )

    movies["tags"] = movies["tags"].fillna("").astype(str)

    # 7. Vectorize
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()

    # 8. Similarity matrix
    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_and_process()

# ----------------- Recommend function -----------------
def recommend(movie, top_n=5):
    if movie not in movies["title"].values:
        return []

    index = movies[movies["title"] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    result = []
    for i, score in distances:
        result.append(movies.iloc[i]["title"])
    return result

# ----------------- Streamlit UI -----------------
st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get similar movie suggestions.")

movie_list = sorted(movies["title"].values)
selected_movie = st.selectbox("Select a movie", movie_list)

top_n = st.slider("How many recommendations do you want?", 3, 10, 5)

if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(selected_movie, top_n=top_n)

    if not recommendations:
        st.error("No recommendations found for this movie.")
    else:
        st.subheader(f"Movies similar to **{selected_movie}**:")
        for i, title in enumerate(recommendations, start=1):
            st.write(f"{i}. ⭐ {title}")

