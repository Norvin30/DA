import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv(r'D:\DA_project_CSV_FILES\IMDB-Movie-Data.csv')

df.dropna(inplace=True)

df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

print("\n🎬 Top 5 Rows:")
print(df.head())

print("\n🔍 Basic Info:")
print(df.info())

print("\n📊 Most Common Genres:")
print(df['genre'].value_counts().head(10))

print("\n🏆 Top Rated Movies:")
top_movies = df.sort_values(by='rating', ascending=False).head(10)
print(top_movies[['title', 'rating']])

print("\n📅 Movies Released Per Year:")
print(df['year'].value_counts().sort_index())

plt.figure(figsize=(10, 6))
df['genre'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Movie Genres')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df['rating'], bins=10, kde=True, color='green')
plt.title('IMDb Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='runtime_(minutes)', y='rating', data=df)
plt.title('Movie Runtime vs IMDb Rating')
plt.xlabel('Runtime (min)')
plt.ylabel('IMDb Rating')
plt.tight_layout()
plt.show()

df['description'] = df['description'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title'].str.lower())

def recommend_movie(title, num_recommendations=5):
    title = title.lower()
    if title not in indices:
        return "❌ Movie not found in database."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    
    recommended = df[['title', 'genre', 'rating']].iloc[movie_indices]
    return recommended


print("\n🎥 Recommendations for 'The Dark Knight':")
print(recommend_movie("The Dark Knight"))

df.to_csv(r'D:/movie_project/cleaned_imdb_movies.csv', index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_imdb_movies.csv'")
