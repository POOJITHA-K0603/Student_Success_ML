import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from .config import *


def engineer(X):
    """Feature engineering (row-wise, so no leakage between train and test)."""
    X = X.copy()
    X["avg_mastery"] = X[MASTERY].mean(axis=1)
    X["min_mastery"] = X[MASTERY].min(axis=1)
    X["engagement"] = X["attendance"] * X["participation"] / 100
    X["distraction_ratio"] = X["screen_hours"] / (X["study_hours"] + 1)
    return X


def make_pipe(model):
    """engineer -> impute -> scale -> model. Imputer/scaler are fitted ONLY on
    the training fold inside the pipeline, which prevents data leakage."""
    return Pipeline([("fe", FunctionTransformer(engineer)),
                     ("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", model)])


def load_split(test_size=0.2):
    df = pd.read_csv(DATA / "students.csv")
    # labels encoded 0/1/2 (= CLASSES order); sklearn's MLP early stopping needs numeric labels
    X, y = df[RAW], df[TARGET].map({c: i for i, c in enumerate(CLASSES)})
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
