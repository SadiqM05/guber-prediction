
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def prepare_data(df):
    df = df.copy()

    # Define target: 0 = Completed, 1 = Cancelled by Customer
    target_map = {
        "Completed": 0,
        "Cancelled by Customer": 1
    }

    df["target"] = df["Booking Status"].map(target_map)

    # Remove booking outcomes outside our prediction scope
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    # Engineer date/time features
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Hour"] = pd.to_datetime(
        df["Time"],
        format="%H:%M:%S",
        errors="coerce"
    ).dt.hour

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

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


def create_preprocessor():
    numeric_features = [
        "Hour", "DayOfWeek", "Month"
    ]

    categorical_features = [
        "Vehicle Type",
        "Pickup Location",
        "Drop Location"
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])

    return preprocessor
