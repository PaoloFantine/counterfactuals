import sys
from pathlib import Path

# PROJECT_ROOT = Path().resolve().parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from src import load_dataset, Counterfactuals

X, X_proc, y = load_dataset("adult", preprocess=True, include_description=True)

# ARTIFACT_DIR = Path("/artifacts")
# ARTIFACT_DIR.mkdir(exist_ok=True)

import joblib

model = joblib.load("artifacts/classification_model.pkl")

instance = X_proc.iloc[0:1]
instance_outcome = y.iloc[0:1]
#instance['outcome'] = instance_outcome.values

cf = Counterfactuals(X=X_proc, y=y, model=model)

pt = cf.get_counterfactuals(
    base=instance,
    n_counterfactuals=5,
    desired_class=1,
    method="prototypes"
    )

ga = cf.get_counterfactuals(
    base=instance,
    n_counterfactuals=5,
    desired_class=1,
    method="genetic",
    one_hot_encoded=["sex_"]
)

pt

ga

check