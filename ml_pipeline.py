import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def clean_data(df):
    df = df.drop_duplicates().copy()
    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())
    return df


def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2,
        random_state=42, stratify=iris.target
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy, X_test


def validate_predictions(predictions, allowed_labels):
    predictions = list(predictions)
    if len(predictions) == 0:
        raise ValueError("Predictions cannot be empty")
    if not set(predictions).issubset(set(allowed_labels)):
        raise ValueError("Prediction contains an invalid class label")
    return True


if __name__ == "_main_":
    model, accuracy, X_test = train_model()
    predictions = model.predict(X_test)
    validate_predictions(predictions, [0, 1, 2])
    print("Model training completed.")
    print("Test accuracy:", round(accuracy, 4))
    print("Number of predictions:", len(predictions))
    print("Model output validation: Passed")