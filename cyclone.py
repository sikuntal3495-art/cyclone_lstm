import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv(
    "ibtracs.last3years.list.v04r01.csv",
    skiprows=[1],
    low_memory=False
)

print("Dataset Shape:", df.shape)



df = df[
    ["SID", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES"]
].copy()


df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

cols = ["LAT", "LON", "WMO_WIND", "WMO_PRES"]

for col in cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df.dropna(
    subset=["SID", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES"],
    inplace=True
)

df = df.sort_values(["SID", "ISO_TIME"])

print("Clean Data:", df.shape)


# Minimum 30 observations per cyclone

counts = df["SID"].value_counts()

valid_sid = counts[counts >= 30].index

df = df[df["SID"].isin(valid_sid)].copy()

print("Total Cyclones:", df["SID"].nunique())



time_step = 8

data = df[cols].values

x = []
y = []
sid_list = []

for sid, cyclone in df.groupby("SID", sort=False):

    cyclone_data = cyclone[cols].values

    if len(cyclone_data) <= time_step:
        continue

    for i in range(time_step, len(cyclone_data)):

        x.append(cyclone_data[i-time_step:i])

        y.append(cyclone_data[i])

        sid_list.append(sid)


X = np.array(x)
Y = np.array(y)

sid_list = np.array(sid_list)

print("X Shape:", X.shape)
print("Y Shape:", Y.shape)



split = int(len(X) * 0.80)

X_train = X[:split]
X_test = X[split:]

y_train = Y[:split]
y_test = Y[split:]

print("X Train:", X_train.shape)
print("X Test:", X_test.shape)

print("Y Train:", y_train.shape)
print("Y Test:", y_test.shape)



scaler = MinMaxScaler()

scaler.fit(
    X_train.reshape(-1, 4)
)

X_train = scaler.transform(
    X_train.reshape(-1, 4)
).reshape(X_train.shape)

X_test = scaler.transform(
    X_test.reshape(-1, 4)
).reshape(X_test.shape)

y_train = scaler.transform(y_train)

y_test = scaler.transform(y_test)



lstm_model = Sequential()

lstm_model.add(Input(shape=(8, 4)))

lstm_model.add(LSTM(64))

lstm_model.add(Dense(4))

lstm_model.compile(
    optimizer="adam",
    loss="mse"
)

lstm_model.summary()


# =========================
# 9. Model Training
# =========================

history = lstm_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.10,
    shuffle=False
)


# =========================
# 10. Prediction
# =========================

pred = lstm_model.predict(X_test)

print("Prediction:")
print(pred[:5])


# =========================
# 11. Original Values
# =========================

pred_original = scaler.inverse_transform(pred)

y_original = scaler.inverse_transform(y_test)

print("Predicted Values:")
print(pred_original[:5])

print("Actual Values:")
print(y_original[:5])


# =========================
# 12. Model Evaluation
# =========================

mse = mean_squared_error(y_original, pred_original)

mae = mean_absolute_error(y_original, pred_original)

r2 = r2_score(y_original, pred_original)

rmse = np.sqrt(mse)

print("MSE:", mse)

print("MAE:", mae)

print("RMSE:", rmse)

print("R2 Score:", r2)


# =========================
# 13. Actual vs Predicted
# =========================

features = ["LAT", "LON", "WIND", "PRESSURE"]

for i in range(4):

    plt.figure(figsize=(10, 5))

    plt.plot(
        y_original[:, i],
        label="Actual"
    )

    plt.plot(
        pred_original[:, i],
        label="Predicted"
    )

    plt.title(features[i] + " Prediction")

    plt.xlabel("Observation")

    plt.ylabel(features[i])

    plt.legend()

    plt.show()


# =========================
# 14. Training Loss
# =========================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()


# =========================
# 15. Save Model
# =========================
# =========================
# 15. Save Model + Scaler
# =========================

import joblib

lstm_model.save("multiple_cyclone_prediction.keras")

joblib.dump(
    scaler,
    "cyclone_scaler.pkl"
)

print("Model Saved Successfully!")
print("Scaler Saved Successfully!")

import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model


# =========================
# 1. Load Model and Scaler
# =========================

model = load_model(
    "multiple_cyclone_prediction.keras"
)

scaler = joblib.load(
    "cyclone_scaler.pkl"
)


# =========================
# 2. User Input Cyclone
# =========================

cyclone_name = input(
    "Enter Cyclone Name: "
)

print("Enter Last 8 Observations")


# =========================
# 3. Input Last 8 Data
# =========================

observations = []

for i in range(8):

    print(f"\nObservation {i+1}")

    lat = float(input("Latitude: "))

    lon = float(input("Longitude: "))

    wind = float(input("Wind Speed: "))

    pressure = float(input("Pressure: "))

    observations.append(
        [lat, lon, wind, pressure]
    )



observations = np.array(
    observations
)

print("\nInput Shape:", observations.shape)



scaled_data = scaler.transform(
    observations
)


X_input = scaled_data.reshape(
    1, 8, 4
)

print("LSTM Input Shape:", X_input.shape)


# =========================
# 7. Predict Next Observation
# =========================

prediction = model.predict(
    X_input,
    verbose=0
)


# =========================
# 8. Original Values
# =========================

prediction_original = scaler.inverse_transform(
    prediction
)


# =========================
# 9. Display Prediction
# =========================

print("\nCyclone Name:", cyclone_name)

print("\nNext Predicted Observation")

print("Latitude:",
      prediction_original[0][0])

print("Longitude:",
      prediction_original[0][1])

print("Wind Speed:",
      prediction_original[0][2])

print("Pressure:",
      prediction_original[0][3])

# =========================
# 7. Predict Future 10 Steps
# =========================

current_sequence = X_input[0].copy()

future_predictions = []

for i in range(10):

    prediction = model.predict(
        current_sequence.reshape(1, 8, 4),
        verbose=0
    )

    next_observation = prediction[0]

    future_predictions.append(
        next_observation
    )

    current_sequence = np.vstack(
        [
            current_sequence[1:],
            next_observation
        ]
    )


# =========================
# 8. Inverse Scaling
# =========================

future_predictions = np.array(
    future_predictions
)

future_original = scaler.inverse_transform(
    future_predictions
)


# =========================
# 9. Print Future Route
# =========================

print("\nCyclone:", cyclone_name)

print("\nFuture 10 Observations:")

for i in range(10):

    print(
        f"\nStep {i+1}"
    )

    print(
        "Latitude:",
        future_original[i][0]
    )

    print(
        "Longitude:",
        future_original[i][1]
    )

    print(
        "Wind:",
        future_original[i][2]
    )

    print(
        "Pressure:",
        future_original[i][3]
    )


# =========================
# 10. Plot Cyclone Route
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    observations[:, 1],
    observations[:, 0],
    marker="o",
    label="Input Route"
)

plt.plot(
    future_original[:, 1],
    future_original[:, 0],
    marker="x",
    linestyle="--",
    label="Predicted Route"
)

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title(
    cyclone_name + " Future Cyclone Route"
)

plt.legend()

plt.grid()

plt.show()    

