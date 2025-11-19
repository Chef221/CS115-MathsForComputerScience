import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("C:\\Users\\Nguyen\\OneDrive\\Documents\\dataset\\drive-download-20250918T142431Z-1-001\\diabetes.csv")
target = "Outcome"
x = data.drop(target, axis = 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
cls = Pipeline(steps= [
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
# cls.fit(x_train, y_train)
# y_predict = cls.predict(x_test)
# print(classification_report(y_test, y_predict))

params = {
    "model__n_estimators": [100, 200, 300],
    "model__criterion": ["gini", "entropy", "log_loss"]
}
grid_search = GridSearchCV(estimator = cls, param_grid = params, cv = 5, verbose = 2, scoring = "recall",  n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
y_predict = grid_search.predict(x_test)
print(classification_report(y_test, y_predict))