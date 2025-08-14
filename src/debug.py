import sys
from pathlib import Path

# PROJECT_ROOT = Path().resolve().parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from src import load_dataset, GA_counterfactuals

X, X_proc, y = load_dataset("adult", preprocess=True, include_description=True)

# ARTIFACT_DIR = Path("/artifacts")
# ARTIFACT_DIR.mkdir(exist_ok=True)

import joblib

model = joblib.load("artifacts/classification_model.pkl")

instance = X_proc.iloc[0:1]
instance_outcome = y.iloc[0:1]
#instance['outcome'] = instance_outcome.values

ga = GA_counterfactuals(X=X_proc, y=y, model=model)

ga.generate_counterfactuals(
    base=instance,
    n_counterfactuals=10,
    desired_class=1,
    )