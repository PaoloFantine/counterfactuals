import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter

from api.helpers import regression_validate_and_serialize
from src import (
    Counterfactuals,
    ModelInfo,
    RegressionCounterfactualRequest,
    RegressionMetrics,
)

router = APIRouter(prefix="/regression", tags=["Regression"])

ARTIFACTS_DIR = Path("artifacts")


# Regression endpoints
@router.get("/model_info/")
async def regression_model_info(request: RegressionCounterfactualRequest):
    return ModelInfo(
        model_type="regression",
        algorithm="XGBoostRegressor",
        dataset="California Housing",
        scope="The California Housing dataset is a standard dataset for regression tasks. It is used to predict the median house value for California districts, based on various features such as location, population, and income. The dataset has been used raw, without preprocessing.",
        output="estimated house price in 100k USD",
    )


@router.get("/metrics/")
async def regression_metrics():
    metrics_path = ARTIFACTS_DIR / "regression_report.json"
    with open(metrics_path, "r") as f:
        metrics_dict = json.load(f)
    return RegressionMetrics(**metrics_dict)


@router.post("/counterfactuals/")
async def regression_counterfactuals(request: RegressionCounterfactualRequest):
    explainer_path = ARTIFACTS_DIR / "regression_explainer.pkl"
    with open(explainer_path, "rb") as f:
        explainer = joblib.load(f)

    cf = Counterfactuals(explainer)

    instance_dict = request.instance.model_dump()

    instance_df = pd.DataFrame([instance_dict])

    cf_df = cf.get_counterfactuals(
        instance_df,
        method=request.method,
        n_counterfactuals=request.n_counterfactuals,
        fix_vars=request.fix_vars,
        lower_limit=request.lower_limit,
        upper_limit=request.upper_limit,
    )

    output = [regression_validate_and_serialize(row) for _, row in cf_df.iterrows()]

    return {
        "message": f"{request.method} counterfactuals for regression",
        "body": output,
    }
