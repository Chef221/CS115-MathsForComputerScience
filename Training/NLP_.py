# NLP project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2, SelectPercentile
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from sklearn.model_selection import GridSearchCV

def filter_location(location):
    res= re.findall(",\s[A-Z]{2}$", location)
    if len(res) != 0:
        return res[0][2:]
    else:
        return location

data = pd.read_excel("C:\\Users\\Nguyen\\OneDrive\\Documents\\dataset\\drive-download-20250918T142431Z-1-001\\final_project.ods", engine = "odf", dtype="str")
data = data.dropna(axis = 0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"
x = data.drop(target, axis = 1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100, stratify = y)

# # balance data
# ros = SMOTEN(random_state=0, k_neighbors = 2, ksampling_strategy={
#     "director_business_unit_leader":500,
#     "specialist":500,
#     "managing_director_small_medium_company":500,
#     "bereichsleiter":1000
# })
# x_train, y_train = ros.fit_resample(x_train, y_train)
# # print(y_train.value_counts())

preprocessor = ColumnTransformer(transformers = [
    ("title_feature", TfidfVectorizer(stop_words="english", ngram_range = (1,1)), "title"),
    ("location_feature", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description_feature", TfidfVectorizer(stop_words="english", ngram_range = (1,2), min_df=0.01, max_df = 0.95), "description"),
    ("function_feature", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry feature", TfidfVectorizer(stop_words="english", ngram_range = (1,1)), "industry")
])

cls = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selector", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier())
])

params = {
    "model__criterion": ["gini", "entropy", "log_loss"],
    "feature_selector__percentile" : [1, 5, 10],
}
grid_search = GridSearchCV(estimator = cls, param_grid = params, cv = 5, scoring = "recall_weighted", verbose = 2, n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
y_predict = grid_search.predict(x_test)
print(classification_report(y_test, y_predict))

# # result = cls.fit_transform(x_train)
# # print(result.shape)
# cls.fit(x_train, y_train)
# y_predict = cls.predict(x_test)
# print(classification_report(y_test, y_predict))