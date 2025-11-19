
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def create_ts_data(data, window_size = 5):
    i = 1
    while i < window_size:
        data["users {}".format(i)] = data["users"].shift(-i)
        i += 1
    data["target"] = data["users"].shift(-i)
    data = data.dropna()
    return data
data = pd.read_csv("C:\\Users\\Nguyen\\OneDrive\\Documents\\dataset\\drive-download-20250918T142431Z-1-001\\Time-series-datasets\\web-traffic.csv")
data["date"] = pd.to_datetime(data["date"])
data = create_ts_data(data)

x = data.drop(["target", "date"], axis = 1)
y = data["target"]

train_ratio = 0.8
num_samples = len(x)
x_train = x[:int(train_ratio * num_samples)]
x_test = x[int(train_ratio * num_samples):]
y_train = y[:int(train_ratio * num_samples)]
y_test = y[int(train_ratio * num_samples):]

model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
print("R2: {}".format(r2_score(y_test, y_pred)))