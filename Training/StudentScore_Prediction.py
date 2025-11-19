import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("C:\\Users\\Nguyen\\OneDrive\\Documents\\dataset\\drive-download-20250918T142431Z-1-001\\StudentScore.xls")

target = "math score"
x = data.drop(target, axis = 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# numerical data : reading score, writing score
# nominal data: race/ethnicity
# ordinal data: gender + parental level of education + lunch + test preparation course
educational_values = x_train["parental level of education"].unique()
gender_values = x_train["gender"].unique()
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
ord_transformer = OrdinalEncoder(categories= [educational_values, gender_values, lunch_values, test_values])

preprocessor = ColumnTransformer(transformers=[
    ("num_transformer", StandardScaler(), ["reading score", "writing score"]),
    ("nom_transformer", OneHotEncoder(), ["race/ethnicity"]),
    ("ord_transformer", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"])
])

reg = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))