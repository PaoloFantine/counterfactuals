ARTIFACTS_DIR="artifacts"
CLASSIFICATION_ARTIFACTS=("classification_report.json" "classification_model.pkl" "categorical_encodings.json")
REGRESSION_ARTIFACTS=("regression_report.json" "xgboost_regressor.pkl")
MISSING=0

for file in "${CLASSIFICATION_ARTIFACTS[@]}"; do
    if [ ! -f "$ARTIFACTS_DIR/$file" ]; then
        MISSING=1
        break
    fi
done

if [ "$MISSING" -eq 1 ]; then
    python -m src.model_training.classification_training --test-size 0.1 --random-state 42
fi

MISSING=0
for file in "${REGRESSION_ARTIFACTS[@]}"; do
    if [ ! -f "$ARTIFACTS_DIR/$file" ]; then
        MISSING=1
        break
    fi
done

if [ "$MISSING" -eq 1 ]; then
    python -m src.model_training.regression_training --test-size 0.1 --random-state 42
fi

uv run uvicorn api.main:app --reload