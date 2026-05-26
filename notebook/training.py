import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# LOAD CLEANED DATASET

df = pd.read_csv(
    "../data/cleaned_insurance.csv"
)

# SELECT FEATURES

X = df[
    [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region"
    ]
]

# TARGET

y = df["charges"]

# TRAIN TEST SPLIT

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# FEATURE SCALING

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

# MODEL TRAINING

model = DecisionTreeRegressor(
    random_state=42
)

model.fit(x_train, y_train)

# PREDICTION

y_pred = model.predict(x_test)

# EVALUATION

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("R2 Score:", r2)

# SAVE MODEL

pickle.dump(
    model,
    open("../models/model.pkl", "wb")
)

# SAVE SCALER

pickle.dump(
    scaler,
    open("../models/scaler.pkl", "wb")
)

print("\nModel and Scaler Saved Successfully")