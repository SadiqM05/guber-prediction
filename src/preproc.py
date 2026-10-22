
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data(df, features=None):
    df = df.copy()

    target_map = {
        "Completed": 0,
        "Cancelled by Customer": 1
    }

    df["target"] = df["Booking Status"].map(target_map)
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Hour"] = pd.to_datetime(
        df["Time"], format="%H:%M:%S", errors="coerce"
    ).dt.hour
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

    if features is None:
        features = [
            "Vehicle Type",
            "Pickup Location",
            "Drop Location",
            "Hour",
            "DayOfWeek",
            "Month"
        ]

    X = df[features].copy()
    y = df["target"].copy()

    return X, y


def create_preprocessor(features=None):
    if features is None:
        features = [
            "Vehicle Type",
            "Pickup Location",
            "Drop Location",
            "Hour",
            "DayOfWeek",
            "Month"
        ]

    numeric_features = [
        col for col in features
        if col in [
            "Hour", "DayOfWeek", "Month",
            "Avg VTAT", "Ride Distance", "Booking Value"
        ]
    ]

    categorical_features = [
        col for col in features
        if col in ["Vehicle Type", "Pickup Location",
                   "Drop Location", "Payment Method"]
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])