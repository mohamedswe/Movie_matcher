import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"C:\Users\mabde\OneDrive\Desktop\Coding\Ml movie matcher\imdb_top_1000 (1).csv")

clean_df=df.drop(columns=['Poster_Link','Certificate','Runtime','No_of_Votes','Gross'])

    # print(clean_df.head)

print(clean_df.columns.to_list)


    # Preprocess the movie overviews and genres
preprocessed_overviews = clean_df['Overview'].fillna('').str.lower().str.replace('[^\w\s]', '')
preprocessed_genres = clean_df['Genre'].fillna('').str.lower().str.replace('[^\w\s]', '')
preprocessed_directors = clean_df['Director'].fillna('').str.lower().str.replace('[^\w\s]', '')

combined_text = preprocessed_overviews + ' ' + preprocessed_genres + '' + preprocessed_directors


vectorizer = TfidfVectorizer(stop_words='english')
movie_vectors = vectorizer.fit_transform(combined_text)

similarity_matrix = cosine_similarity(movie_vectors)

    # Function to get movie recommendations
def get_recommendations(movie_title, df, similarity_matrix, top_n=5):
    # Find the index of the target movie in the DataFrame
    movie_index = df[df['Series_Title'] == movie_title].index[0]

    # Get the similarity scores for the target movie
    similarity_scores = similarity_matrix[movie_index]

    # Get the indices of the top similar movies
    top_indices = similarity_scores.argsort()[::-1][1:top_n+1]

    # Get the recommended movie titles
    recommended_movies = df.iloc[top_indices]['Series_Title'].tolist()

    return recommended_movies

# Example usage
target_movie = "The Notebook"
recommendations = get_recommendations(target_movie, df, similarity_matrix)
print(f"Recommended movies for '{target_movie}':")
for movie in recommendations:
    print(movie)
