"""
Train an XGBoost classifier on the Adult dataset,
save the model and a classification report to `artifacts/`.
Run via:
    python -m main.model_training_scripts.classification_training --test-size 0.1 --random-state 42
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_loader import load_dataset

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)


def train(test_size: float, random_state: int) -> None:
    # Load data and preprocess it
    logging.info("Loading dataset …")
    X, X_proc, y = load_dataset("adult", preprocess=True)
    X_proc["education_num"] = X_proc["education_num"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=test_size, random_state=random_state
    )

    # Train model
    logging.info("Training XGBoost …")
    clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        enable_categorical=True,
    )
    clf.fit(X_train, y_train)

    # Evaluate model
    logging.info("Evaluating …")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # store evaluation metrics
    logging.info("Generating classification report …")
    with open(ARTIFACT_DIR / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Saved classification_report.json")

    # Serialize model
    joblib.dump(clf, ARTIFACT_DIR / "classification_model.pkl")
    logging.info("Saved classification_model.pkl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.test_size, args.random_state)
