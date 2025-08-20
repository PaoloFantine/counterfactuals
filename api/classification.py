import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter

from api.helpers import encode_input, serialize_df_with_pydantic
from src import (
    ClassificationCounterfactualRequest,
    ClassificationReport,
    Counterfactuals,
    ModelInfo,
    load_dataset,
)

router = APIRouter(prefix="/classification", tags=["Classification"])

ARTIFACTS_DIR = Path("artifacts")


# Classification endpoints
@router.get("/model_info/")
async def classification_model_info():
    return ModelInfo(
        model_type="classification",
        algorithm="XGBoostClassifier",
        dataset="adult",
        scope="The Adult dataset is a standard dataset for classification tasks. It is used to predict whether an individual earns more than $50,000 per year based on various features such as age, education, and occupation. The dataset has been preprocessed to ordinally encode categorical features and remove the unencoded ones.",
        output=str({"1": ">50K", "0": "<=50K"}),
    )


@router.get("/report/")
async def classification_report():
    report_path = ARTIFACTS_DIR / "classification_report.json"
    with open(report_path, "r") as f:
        report_dict = json.load(f)
    return ClassificationReport(root=report_dict).model_dump()


@router.post("/counterfactuals/")
async def classification_counterfactuals(request: ClassificationCounterfactualRequest):

    _, X_proc, y = load_dataset("adult", preprocess=True)
    model = joblib.load(ARTIFACTS_DIR / "classification_model.pkl")

    cf = Counterfactuals(X=X_proc, y=y, model=model)

    # Encode categorical values
    encoded_instance = encode_input(request.instance)
    # Convert to a single-row DataFrame
    instance_df = pd.DataFrame([encoded_instance])

    outcome = instance_df["outcome"].iloc[0]

    expected_columns = X_proc.columns  # This preserves both order and names
    instance_df = instance_df[expected_columns]
    instance_df["outcome"] = outcome

    n_counterfactuals = request.n_counterfactuals
    method = request.method
    fix_vars = request.fix_vars if request.fix_vars is not None else []
    desired_outcome = request.desired_outcome
    one_hot_encoded = request.one_hot_encoded

    cf_df = cf.get_counterfactuals(
        instance_df,
        method=method,
        n_counterfactuals=n_counterfactuals,
        fix_vars=fix_vars,
        desired_class=desired_outcome,
        one_hot_encoded=one_hot_encoded,
    )

    json_output = serialize_df_with_pydantic(cf_df)

    return {
        "message": f"{method} Counterfactuals for classification",
        "body": json_output,
    }
