from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "Crop_recommendation.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading dataset from: {data_path}")
    data = pd.read_csv(data_path)

    # Encode labels
    le = LabelEncoder()
    data["label"] = le.fit_transform(data["label"])  # type: ignore[index]

    # Split features/target
    X = data.drop("label", axis=1)
    y = data["label"]

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Pipeline (scaler + RF)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
        ]
    )

    # Hyperparameter tuning
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [6, 7],
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=skf, scoring="accuracy", n_jobs=-1
    )
    print("Training (GridSearchCV)... this may take a couple of minutes...")
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")

    # Persist artifacts
    out_dir = root / "model_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, out_dir / "model.joblib")
    joblib.dump(le, out_dir / "le.joblib")

    # Optional: human-readable mapping
    label_mapping = {int(i): cls for i, cls in enumerate(le.classes_)}
    with (out_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    print(f"Saved model artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

