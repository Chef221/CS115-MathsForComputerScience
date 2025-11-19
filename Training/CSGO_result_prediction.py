import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
data = pd.read_csv("C:\\Users\\Nguyen\\OneDrive\\Documents\\dataset\\drive-download-20250918T142431Z-1-001\\csgo.csv")

target = "result"
x = data.drop(["result", "day", "month", "year", "date", "team_a_rounds", "team_b_rounds", "wait_time_s"], axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# nominal value: map
# numerical value: match_time_s + ping + kills + assists + deaths + mvps + hs_percent + points

preprocessor = ColumnTransformer(transformers=[
    ("nom_transformer", OneHotEncoder(), ["map"]),
    ("scaler", StandardScaler(), ["match_time_s", "ping", "kills", "assists", "deaths", "mvps", "hs_percent", "points"])
])
cls = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])
cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))