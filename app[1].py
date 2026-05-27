import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
df = pd.read_csv("Dataset .csv")

# Remove missing values
df = df.dropna()

# Use only important columns
df = df[['Restaurant Name', 'Cuisines']]

# Remove duplicates
df = df.drop_duplicates()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Cuisines'])

# Cosine Similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommendation Function
def recommend_restaurant(cuisine):

    cuisine = cuisine.lower()

    matching_restaurants = df[df['Cuisines'].str.lower().str.contains(cuisine)]

    return matching_restaurants.head(10)

# Streamlit UI
st.title("Restaurant Recommendation System")

st.write("Get restaurant recommendations based on cuisine preferences")

user_input = st.text_input("Enter Preferred Cuisine")

if st.button("Recommend"):

    recommendations = recommend_restaurant(user_input)

    if not recommendations.empty:
        st.write(recommendations[['Restaurant Name', 'Cuisines']])
    else:
        st.warning("No matching restaurants found.")
