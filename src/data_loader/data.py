# Script to load datasets for demo purposes
import json
from pathlib import Path
from pprint import pprint

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


ADULT_COLUMN_DESCRIPTION = {
    "age": "Age of the individual",
    "workclass": "Represents the employment status",
    "fnlwgt": "Final weight — estimation of census population",
    "education": "Highest level of education achieved",
    "education-num": "Education level as an integer",
    "marital-status": "Marital status",
    "occupation": "Occupation",
    "relationship": "Relationship status",
    "race": "Race",
    "sex": "Gender",
    "capital-gain": "Capital gain in USD",
    "capital-loss": "Capital loss in USD",
    "hours-per-week": "Average working hours per week",
    "native-country": "Country of origin",
    "target": "Income level, '>50K' or '<=50K'",
}
# remove education as it is redundant with education-num,
# and occupation as it would be by far the biggest predictor of income
# fnlwgt is also not useful for prediction as it is a sampling artifact
ADULT_DROP_COLUMNS = ["education", "occupation", "fnlwgt"]

CALIFORNIA_HOUSING_COLUMN_DESCRIPTION = {
    "MedInc": "Median income in block group",
    "HouseAge": "Median house age in block group",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Block group population",
    "AveOccup": "Average house occupancy (household size)",
    "Latitude": "Block group latitude",
    "Longitude": "Block group longitude",
    "target": "MedHouseVal, Median house value in 100k USD",
}


def load_dataset(name, preprocess=False, include_description=False):
    if name == "adult":
        ds = fetch_openml("adult", version=2, as_frame=True)
        print(ds.data.columns)
        ds.data.rename(
            columns={
                "marital-status": "marital_status",
                "native-country": "native_country",
                "education-num": "education_num",
                "hours-per-week": "hours_per_week",
                "capital-gain": "capital_gain",
                "capital-loss": "capital_loss",
            },
            inplace=True,
        )

        X, y = ds.data, (ds.target == ">50K").astype(int)
        if preprocess:
            X_clean = X.copy().drop(columns=ADULT_DROP_COLUMNS, errors="ignore")
            preprocessor = build_preprocessor(X_clean)
            X_proc = preprocessor.fit_transform(X_clean)

            # store encodings for later use
            store_cat_encodings(preprocessor)

            X_proc = pd.DataFrame(X_proc, columns=preprocessor.get_feature_names_out())

            # Convert categorical features to int
            cat_pipeline_features = X_proc.columns[
                X_proc.columns.str.startswith("cat__")
            ]
            X_proc[cat_pipeline_features] = X_proc[cat_pipeline_features].astype(int)

            # take care of naming
            X_proc.columns = X_proc.columns.str.replace("num__", "", regex=False)
            X_proc.columns = X_proc.columns.str.replace("cat__", "", regex=False)
            result = (X, X_proc, y)
        else:
            result = (X, None, y)

        if include_description:
            pprint(ADULT_COLUMN_DESCRIPTION)
            
        # Ensure specified columns are of type int if they exist
        cols_to_int = [
            "education_num",
            "age",
        ]
        for col in cols_to_int:
            X_proc[col] = X_proc[col].astype(int)
                
        # Convert columns to categorical
        cols_to_cat = [
            "workclass",
            "marital_status",
            "relationship",
            "race",
            "sex",
            "native_country",
            ]
        
        for col in cols_to_cat:
            X_proc[col] = X_proc[col].astype('category')
                
        return result
    elif name == "california_housing":
        ds = fetch_california_housing(as_frame=True)
        X, y = ds.data, ds.target

        if include_description:
            pprint(CALIFORNIA_HOUSING_COLUMN_DESCRIPTION)
        return (X, y)


def build_preprocessor(X):
    "preprocessor for adult dataset"
    # Identify column types
    cat_cols = [
        "workclass",
        "marital_status",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    num_cols = [
        "age",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]

    # Pipelines
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)]
    )

    return preprocessor


def store_cat_encodings(preprocessor):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    cat_mapping = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            encoder = transformer.named_steps["encoder"]
            for col, cats in zip(columns, encoder.categories_):
                # Convert each category to a string (in case of numpy types)
                cat_mapping[col] = {str(cat): int(idx) for idx, cat in enumerate(cats)}

    # Write to JSON file
    path = ARTIFACT_DIR / "categorical_encodings.json"
    try:
        with open(path, "w") as f:
            json.dump(cat_mapping, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not store categorical encodings: {e}")
    print(
        f"Stored categorical mappings in: {ARTIFACT_DIR / 'categorical_encodings.json'}"
    )