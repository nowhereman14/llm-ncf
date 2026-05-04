import pandas as pd
import numpy as np

#*****************LOAD DATA*******************
ratings = pd.read_csv('ml-25m/ratings.csv')
movies = pd.read_csv('ml-25m/movies.csv')
movies['genres'] = movies['genres'].replace('(no genres listed)', '')
genome_scores = pd.read_csv('ml-25m/genome-scores.csv')
genome_tags = pd.read_csv('ml-25m/genome-tags.csv')
tags = pd.read_csv('ml-25m/tags.csv')

sample_users = ratings['userId'].drop_duplicates().sample(frac=0.05, random_state=42)
ratings = ratings[ratings['userId'].isin(sample_users)]

sample_movies = ratings['movieId'].unique()
genome_scores = genome_scores[genome_scores['movieId'].isin(sample_movies)]
tags = tags[tags['movieId'].isin(sample_movies)]

tags_grouped = (tags.dropna(subset=['tag'])
            .drop_duplicates(subset=['movieId', 'tag'])
            .groupby('movieId')['tag']
            .apply(lambda x: ' '.join(x))
            .reset_index())

#*****************MOVIE FEATURES*******************
genome = genome_scores.merge(genome_tags, on='tagId')
top_genome = (genome.sort_values('relevance', ascending=False)
                    .groupby('movieId')
                    .head(20)
                    .groupby('movieId')['tag']
                    .apply(lambda x: ' '.join(x))
                    .reset_index()
                    .rename(columns={'tag': 'genome_tags'}))

user_tags = tags_grouped.merge(top_genome, on='movieId', how='outer')
user_tags['user_tags'] = (
    user_tags['tag'].fillna('') + ' ' +
    user_tags['genome_tags'].fillna('')
)
user_tags = user_tags[['movieId', 'user_tags']]
user_tags = user_tags.rename(columns={'user_tags': 'processed_features'})

movies_features = movies.merge(user_tags, on='movieId', how='left')
movies_features['processed_features'] = (
    movies_features['title'] + ' | ' +
    movies_features['genres'] + ' | ' +
    movies_features['processed_features']
)
movies_features = movies_features[['movieId', 'processed_features']]
movies_features['processed_features'] = movies_features['processed_features'].fillna('')

#*****************BUILD POSITIVES*******************
ratings['rating_text'] = pd.cut(
    ratings['rating'],
    bins=[0, 2.5, 3.5, 5],
    labels=['poorly rated', 'average', 'highly rated']
)
ratings = ratings[['userId', 'movieId', 'rating_text']]

movies_reviews = ratings.merge(movies_features, on='movieId', how='left')
movies_reviews['processed_features'] = (
    movies_reviews['processed_features'] + ' | ' +
    movies_reviews['rating_text'].astype(str).replace('nan', '')
)
print(movies_reviews.head())
movies_reviews['interaction'] = 1
positives = movies_reviews[['userId', 'movieId', 'interaction', 'processed_features']]
print(positives.info())

#*****************BUILD NEGATIVES*******************
all_movies = movies_reviews['movieId'].unique()
negative_samples = []
for user_id, group in movies_reviews.groupby('userId'):
    seen = set(group['movieId'].values)
    unseen = list(set(all_movies) - seen)
    n_samples = len(seen)
    negatives = np.random.choice(unseen, size=n_samples, replace=False)

    for movie_id in negatives:
        negative_samples.append({'userId': user_id, 
                                'movieId': movie_id, 
                                'interaction': 0})
    
negatives = pd.DataFrame(negative_samples)
print(negatives.info())

#*****************MERGING POSITIVE AND NEGATIVE*******************
negatives = negatives.merge(movies_features[['movieId', 'processed_features']], 
                            on='movieId', 
                            how='left')
df = pd.concat([positives, negatives], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

print(df.info())
print(df.head())

#*****************SAVE TO CSV*******************
df.to_csv('dataset.csv', index=False)