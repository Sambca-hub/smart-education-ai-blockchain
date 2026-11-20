import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from evaluation import evaluate_models


def train_models(
    df,
    target,
    threshold=10,
    test_size=0.20,
    excluded_columns=None,
):
    """Train and compare three classification models."""
    if target not in df.columns:
        raise ValueError("Target column not found.")

    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test size must be between 10% and 40%.")

    excluded_columns = excluded_columns or []
    working = df.copy().replace([np.inf, -np.inf], np.nan)
    working = working.dropna(subset=[target])

    if pd.api.types.is_numeric_dtype(working[target]):
        y = pd.Series(
            np.where(
                working[target].astype(float) >= threshold,
                "On Track",
                "At Risk",
            ),
            index=working.index,
        )
        numeric_target = True
    else:
        y = working[target].astype(str)
        numeric_target = False

    features = [
        col for col in working.columns
        if col != target and col not in excluded_columns
    ]
    X = working[features].copy()
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("No usable input features remain.")
    if y.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    if len(X) < 10:
        raise ValueError("At least 10 usable records are required.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_counts = pd.Series(y_encoded).value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Every target class needs at least two records. "
            "Consider combining rare classes."
        )

    # Ensure the test set can contain at least one example of every class.
    n_classes = len(encoder.classes_)
    n_test = int(np.ceil(len(X) * test_size))
    n_train = len(X) - n_test
    if n_test < n_classes or n_train < n_classes:
        raise ValueError(
            "The dataset is too small for this number of classes and "
            "test percentage. Add more records or reduce the number of classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )

    numerical = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = [col for col in X_train.columns if col not in numerical]
    transformers = []

    if numerical:
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numerical_pipeline, numerical))

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))

    preprocessor = ColumnTransformer(transformers=transformers)

    algorithms = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
    }

    trained_models = {}
    predictions = {}

    for name, algorithm in algorithms.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", algorithm),
        ])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        predictions[name] = pipeline.predict(X_test)

    metrics = evaluate_models(y_test, predictions)

    return {
        "models": trained_models,
        "metrics": metrics,
        "encoder": encoder,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "features": X.columns.tolist(),
        "target": target,
        "numeric_target": numeric_target,
        "threshold": threshold,
    }


def predict_students(model, df, features, encoder):
    """Predict class labels for records in a compatible dataframe."""
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(
            "Prediction data is missing required columns: "
            + ", ".join(missing)
        )

    X = df[features].copy().replace([np.inf, -np.inf], np.nan)
    predictions = model.predict(X)
    return encoder.inverse_transform(predictions)
