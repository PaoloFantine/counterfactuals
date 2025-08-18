from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, RootModel, field_validator


# model infos
class ModelInfo(BaseModel):
    model_type: str = Field(
        ..., description="Type of model, e.g. classification or regression"
    )
    algorithm: str = Field(
        ..., description="Algorithm used for the model, e.g. XGBoostClassifier"
    )
    dataset: str = Field(..., description="Dataset used for training the model")
    scope: str = Field(
        ..., description="Scope of the model, describing the dataset and its features"
    )
    output: str = Field(
        ...,
        description="Output variable of the model, if applicable; classes for classification models",
    )


# model reports
class Metrics(BaseModel):
    precision: float = Field(..., description="Precision of the model TP/ (TP + FP)")
    recall: float = Field(..., description="Recall of the model TP/ (TP + FN)")
    f1_score: float = Field(
        alias="f1-score",
        description="F1 score of the model 2 * (precision * recall) / (precision + recall)",
    )
    support: float = Field(
        ..., description="Support of the model, number of true instances for each label"
    )

    model_config = {"populate_by_name": True}


class ClassificationReport(RootModel):
    root: Dict[Union[str, int], Union[Metrics, float]]

    def __getitem__(self, item):
        return self.__root__[item]


class RegressionMetrics(BaseModel):
    mse: float = Field(..., description="Mean Squared Error of the model")
    mae: float = Field(..., description="Mean Absolute Error of the model")
    r2: float = Field(..., description="R-squared of the model")


# Model inputs
class MaritalStatus(str, Enum):
    married_civ_spouse = "Married-civ-spouse"
    divorced = "Divorced"
    never_married = "Never-married"
    separated = "Separated"
    widowed = "Widowed"
    married_spouse_absent = "Married-spouse-absent"
    married_af_spouse = "Married-AF-spouse"


class Relationship(str, Enum):
    husband = "Husband"
    wife = "Wife"
    own_child = "Own-child"
    unmarried = "Unmarried"
    not_in_family = "Not-in-family"
    other_relative = "Other-relative"


class Workclass(str, Enum):
    private = "Private"
    self_emp_not_inc = "Self-emp-not-inc"
    self_emp_inc = "Self-emp-inc"
    federal_gov = "Federal-gov"
    local_gov = "Local-gov"
    state_gov = "State-gov"
    without_pay = "Without-pay"
    never_worked = "Never-worked"


class Sex(str, Enum):
    male = "Male"
    female = "Female"


class Race(str, Enum):
    white = "White"
    black = "Black"
    asian_pac_islander = "Asian-Pac-Islander"
    amer_indian_eskimo = "Amer-Indian-Eskimo"
    other = "Other"


class NativeCountry(str, Enum):
    united_states = "United-States"
    mexico = "Mexico"
    philippines = "Philippines"
    germany = "Germany"
    canada = "Canada"
    cambodia = "Cambodia"
    china = "China"
    columbia = "Columbia"
    cuba = "Cuba"
    dominican = "Dominican-Republic"
    ecuador = "Ecuador"
    salvadorean = "El-Salvador"
    engliand = "England"
    france = "France"
    greece = "Greece"
    guatemala = "Guatemala"
    haiti = "Haiti"
    holland = "Holand-Netherlands"
    honduras = "Honduras"
    hong = "Hong"
    hungary = "Hungary"
    india = "India"
    iran = "Iran"
    ireland = "Ireland"
    italy = "Italy"
    jamaica = "Jamaica"
    japan = "Japan"
    laos = "Laos"
    nicaragua = "Nicaragua"
    outlying_us = "Outlying-US(Guam-USVI-etc)"
    peru = "Peru"
    poland = "Poland"
    portugal = "Portugal"
    puertorican = "Puerto-Rico"
    scotland = "Scotland"
    south = "South"
    taiwan = "Taiwan"
    thailand = "Thailand"
    trinidad_tobago = "Trinadad&Tobago"
    vietnam = "Vietnam"
    yugoslavia = "Yugoslavia"


class ClassificationInput(BaseModel):
    age: int
    education_num: int
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    marital_status: MaritalStatus
    relationship: Relationship
    workclass: Workclass
    sex: Sex
    race: Race
    native_country: NativeCountry
    outcome: float


class ClassificationCounterfactualRequest(BaseModel):
    instance: ClassificationInput
    n_counterfactuals: Optional[int] = 3
    method: Literal["prototypes", "genetic"] = "genetic"
    fix_vars: Optional[List[str]] = None
    desired_outcome: int

    @field_validator("fix_vars")
    def check_fix_vars(cls, fix_vars):
        if fix_vars is None:
            return fix_vars

        allowed_cols = {
            field.alias or name
            for name, field in ClassificationInput.model_fields.items()
        } - {"outcome"}

        for v in fix_vars:
            if v not in allowed_cols:
                raise ValueError(
                    f"fix_vars item '{v}' is not a valid column. Allowed: {allowed_cols}"
                )
        return fix_vars


class RegressionInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    outcome: float


class RegressionCounterfactualRequest(BaseModel):
    instance: RegressionInput
    n_counterfactuals: Optional[int] = 3
    method: Literal["prototypes", "genetic"] = "genetic"
    fix_vars: Optional[List[str]] = None
    lower_limit: float
    upper_limit: float

    @field_validator("fix_vars")
    def check_fix_vars(cls, fix_vars):
        if fix_vars is None:
            return fix_vars

        allowed_cols = set(RegressionInput.model_fields.keys())
        for v in fix_vars:
            if v not in allowed_cols:
                raise ValueError(
                    f"fix_vars item '{v}' is not a valid column. Allowed: {allowed_cols}"
                )
        return fix_vars
