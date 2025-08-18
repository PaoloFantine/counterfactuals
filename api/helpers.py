import json
from pathlib import Path
from typing import List

import pandas as pd

from src import ClassificationInput, RegressionInput

ARTIFACTS_DIR = Path("artifacts")

ENCODING_PATH = Path(ARTIFACTS_DIR / "categorical_encodings.json")

with open(ENCODING_PATH) as f:
    CAT_ENCODINGS = json.load(f)

INVERSE_CAT_ENCODINGS = {
    col: {v: k for k, v in mapping.items()} for col, mapping in CAT_ENCODINGS.items()
}


def convert(row):
    # Cast all int-like fields to int if they're float
    for field, model_field in ClassificationInput.model_fields.items():
        alias = model_field.alias or field
        if alias in row:
            val = row[alias]
            if isinstance(val, float) and val.is_integer():
                row[alias] = int(val)
    return ClassificationInput.model_validate(row).model_dump(by_alias=True)


def serialize_df_with_pydantic(df: pd.DataFrame) -> List[dict]:
    return [convert(decode_output(row.to_dict())) for _, row in df.iterrows()]


def encode_input(raw_input: ClassificationInput) -> dict:
    data = raw_input.model_dump()

    for col, mapping in CAT_ENCODINGS.items():
        if col in data and data[col] is not None:
            label = data[col].value
            if label not in mapping:
                raise ValueError(f"Invalid value '{label}' for field '{col}'")
            data[col] = mapping[label]

    return data


def decode_output(data: dict) -> dict:
    for col, mapping in INVERSE_CAT_ENCODINGS.items():
        if col in data and data[col] in mapping:
            data[col] = mapping[data[col]]
    return data


def regression_validate_and_serialize(row: pd.Series) -> dict:
    model = RegressionInput.model_validate(row.to_dict())
    return model.model_dump()
