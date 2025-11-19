import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

def format_title(title):
    return title.replace("|", " ").replace("-", "")
def get_recommendations(input_movie, cosine_sim_dense, top_k = 20):
    result = (
        cosine_sim_dense.loc[input_movie, :]
                .drop(input_movie)
                .sort_values(ascending=False)
                .head(top_k)
                .reset_index()
    )
    result.rename(columns = {"index": "movie"}, inplace = True)
    return result
data = pd.read_csv("C:\\Users\\Nguyen\\OneDrive\\Documents\\dataset\\drive-download-20250918T142431Z-1-001\\movie_data\\movies.csv", encoding = "latin-1", sep = "\t", usecols = ["title", "genres"])
data["genres"] = data["genres"].apply(format_title)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["genres"])
tfidf_matrix_dense = pd.DataFrame(vectorizer.fit_transform(data["genres"]).todense(), index = data["title"], columns = vectorizer.get_feature_names_out())

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_dense = pd.DataFrame(cosine_sim, index = data["title"], columns = data["title"])

top20 = get_recommendations(data["title"], cosine_sim_dense, 20)
print(top20)

