import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans


class ClusterAdder(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X, y=None):
        self.kmeans.fit(X[["latitude", "longitude"]])
        return self

    def transform(self, X):
        X = X.copy()
        X["location_cluster"] = self.kmeans.predict(X[["latitude", "longitude"]])
        return X


class BedroomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        X = X.copy()
        X["rooms_bin"] = pd.qcut(X["total_rooms"], q=self.n_bins, duplicates="drop")
        _, self.bins = pd.qcut(X["total_rooms"], q=self.n_bins, retbins=True, duplicates="drop")
        self.medians = X.groupby("rooms_bin")["total_bedrooms"].median()
        self.global_median = X["total_bedrooms"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X["rooms_bin"] = pd.cut(X["total_rooms"], bins=self.bins, include_lowest=True)

        for b in self.medians.index:
            mask = (X["rooms_bin"] == b) & (X["total_bedrooms"].isna())
            X.loc[mask, "total_bedrooms"] = self.medians[b]

        X["total_bedrooms"] = X["total_bedrooms"].fillna(self.global_median)
        X = X.drop(columns="rooms_bin")
        return X


def add_features(X):
    X = X.copy()
    X["rooms_per_household"] = X["total_rooms"] / X["households"]
    X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
    X["population_per_household"] = X["population"] / X["households"]
    X["income_per_person"] = X["median_income"] / X["population"]
    X["bedrooms_per_person"] = X["total_bedrooms"] / X["population"]
    X["rooms_per_person"] = X["total_rooms"] / X["population"]

    city_lat = 37.7749
    city_lon = -122.4194

    X["distance_to_city"] = np.sqrt(
        (X["latitude"] - city_lat) ** 2 +
        (X["longitude"] - city_lon) ** 2
    )

    X["distance_to_coast"] = abs(X["longitude"] + 122)
    return X


feature_adder = FunctionTransformer(add_features, validate=False)

num_cols = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "latitude",
    "longitude",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "distance_to_city",
    "distance_to_coast",
    "income_per_person",
    "rooms_per_person",
    "bedrooms_per_person",
]

cat_cols = [
    "ocean_proximity",
    "location_cluster",
]

preprocessor = Pipeline([
    ("bedrooms_imputer", BedroomImputer(n_bins=10)),
    ("class_adder", ClusterAdder()),
    ("add_features", feature_adder),
    ("column_transformer", ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])),
])