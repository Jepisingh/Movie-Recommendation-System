import streamlit as st
import pandas as pd
import ast
import requests
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize PorterStemmer
ps = PorterStemmer()

# --- Helper Functions ---
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Function to fetch movie poster from TMDB API (With Debugging)
def fetch_poster(movie_id):
    # YOUR API KEY
    api_key = "6ce5ae25fc9f48a70484d256d893b4dc"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    
    try:
        # Added timeout to prevent hanging
        response = requests.get(url, timeout=5)
        
        # Check if the API call was successful
        if response.status_code == 200:
            data = response.json()
            # Check if the movie actually has a poster path
            if 'poster_path' in data and data['poster_path']:
                return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
            else:
                # API worked, but this specific movie has no image
                return "https://via.placeholder.com/500x750?text=No+Poster+Available"
        else:
            # API Key error or limit reached
            return f"https://via.placeholder.com/500x750?text=API+Error+{response.status_code}"
            
    except Exception as e:
        # Internet or Connection issues
        return "https://via.placeholder.com/500x750?text=Connection+Error"

# Page configuration
st.set_page_config(page_title="CineMatch - Movie Recommender", layout="wide", page_icon="🎬")

# Custom CSS for UI
st.markdown("""
    <style>
    .stApp { background-color: #141414; color: #ffffff; }
    h1 { color: #E50914; font-weight: 800; text-transform: uppercase; text-align: center; }
    h3 { color: #f5f5f5; }
    .stButton>button { background-color: #E50914; color: white; border-radius: 4px; border: none; font-weight: bold; text-transform: uppercase; width: 100%; }
    .stButton>button:hover { background-color: #f40612; }
    div[data-testid="stImage"] img { border-radius: 10px; transition: transform 0.3s; }
    div[data-testid="stImage"] img:hover { transform: scale(1.05); }
    .rating { color: #f5c518; font-weight: bold; font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_and_process_data():
    movies = pd.read_csv('tmdb_5000_movies.csv.zip')
    credits = pd.read_csv('tmdb_5000_credits.csv.zip')
    
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average']]
    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)
    
    def convert(obj):
        l = []
        try:
            for i in ast.literal_eval(obj):
                l.append(i['name'])
        except: pass
        return l

    def convert3(obj):
        l = []
        counter = 0
        try:
            for i in ast.literal_eval(obj):
                if counter != 3:
                    l.append(i['name'])
                    counter += 1
                else: break
        except: pass
        return l
        
    def fetch_director(obj):
        l = []
        try:
            for i in ast.literal_eval(obj):
                if i["job"] == 'Director':
                    l.append(i['name'])
                    break
        except: pass
        return l

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])
    
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    new_df = movies[['movie_id', 'title', 'tags', 'vote_average']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    new_df['tags'] = new_df['tags'].apply(stem)
    
    return new_df

@st.cache_resource
def get_similarity_matrix(new_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    return cosine_similarity(vectors)

# --- App Logic ---
st.title("🎬 CineMatch Pro")

try:
    with st.spinner('Loading Movie Database...'):
        new_df = load_and_process_data()
        similarity = get_similarity_matrix(new_df)
except FileNotFoundError:
    st.error("CSV files not found.")
    st.stop()

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_posters = []
    recommended_ratings = []
    
    for i in movies_list:
        movie_id = new_df.iloc[i[0]].movie_id
        recommended_movies.append(new_df.iloc[i[0]].title)
        recommended_ratings.append(new_df.iloc[i[0]].vote_average)
        recommended_posters.append(fetch_poster(movie_id))
        
    return recommended_movies, recommended_posters, recommended_ratings

selected_movie = st.selectbox("Type or select a movie you like:", new_df['title'].values)

if st.button('Show Recommendations'):
    names, posters, ratings = recommend(selected_movie)
    st.write("---")
    st.subheader(f"If you liked '{selected_movie}', you'll love:")
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            # Using the updated parameter to avoid warnings
            st.image(posters[idx], use_container_width=True)
            st.markdown(f"**{names[idx]}**")
            st.markdown(f"<span class='rating'>⭐ {ratings[idx]}/10</span>", unsafe_allow_html=True)

