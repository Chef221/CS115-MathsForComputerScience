import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils.estimator_checks import check_estimator_repr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def create_ts_data(data, window_size = 5):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis = 0)
    return data

data = pd.read_csv("C:\\Users\\Nguyen\\OneDrive\\Documents\\dataset\\drive-download-20250918T142431Z-1-001\\Time-series-datasets\\co2.csv")
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()
data = create_ts_data(data)

x = data.drop(["target", "time"], axis = 1)
y = data["target"]

train_ratio = 0.8
num_samples = len(x)
x_train = x[:int(train_ratio * num_samples)]
y_train = y[:int(train_ratio * num_samples)]
x_test = x[int(train_ratio * num_samples):]
y_test = y[int(train_ratio * num_samples):]

model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)


print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))

# Visualization
fig, ax = plt.subplots()
ax.plot(data["time"][:int(train_ratio * num_samples)], data["co2"][:int(train_ratio * num_samples)], label = "Test")
ax.plot(data["time"][int(train_ratio * num_samples):], data["co2"][int(train_ratio * num_samples):], label = "Train")
ax.plot(data["time"][int(num_samples * train_ratio):], y_predict, label = "prediction")
ax.set_xlabel("Year")
ax.set_ylabel("CO2")
ax.legend()
ax.grid()


current_data = [380.5, 390, 390.2, 390.4, 393]
for i in range(10):
    prediction = model.predict([current_data])[0]
    print("CO2 in week {} is {}".format(i + 1, prediction))
    current_data.append(prediction)
    current_data = current_data[1:]