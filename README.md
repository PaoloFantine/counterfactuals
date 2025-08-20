# counterfactuals
Counterfactual explanations made simple: prototype search & genetic algorithm-based methods for interpretable ML

A Python toolkit for generating interpretable counterfactual explanations for both classification and regression models.
Built on top of scikit-learn compatible models and the explainerdashboard package, this project enables model interpretability and counterfactual prototype search as well as genetic algorithm-based explanations.

A demo of the functionality can be found in `notebooks/basic_demo.ipynb`

## Features

- **Prototype-based counterfactual search** using KDTree - looking for data instances close to the examined one in feature space that show the desired behavior

- **Genetic algorithm counterfactual generation** for actionable, realistic changes - Nondominated Sorting Genetic Algorithm II (NSGA-II) with multi-objective optimization [https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf](https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf), [https://arxiv.org/pdf/2004.11165](https://arxiv.org/pdf/2004.11165)

- Support for both (multiclass) **classification and regression** tasks

- Utility scripts for loading sample datasets, training models, and saving sample artifacts

- **Local API** to showcase an example on how to serve counterfactuals interactively

## Getting started

0. Clone the repo:
```sh
git clone git@github.com:PaoloFantine/counterfactuals.git
cd counterfactuals
```
Dependencies are managed using uv.

The `pyproject.toml` contains all the needed dependencies. For development and runnning the code, using a virtual environment is recommended.

So one should install uv (Linux example: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

1. Create the virtual environment and activate it:
```sh
uv venv .venv
source .venv/bin/activate
```

2. Create the uv.lock and install the necessary dependencies:
```sh
uv lock
uv sync --locked --active
```

3. Check out the demo at `./notebooks/basic_demo.ipynb`

## API Usage

One can test and play around with the code in a jupyter notebook. A demo of the package functionality can be found in `./notebooks/basic_demo.ipynb` 

Alternatively, using the provided API is an option. It can be started locally by running:

```sh
uv run uvicorn api.main:app --reload
```

Before trying out the demos or running the API, some artifacts are needed. These can be generated, for demo purposes, by running:

```sh
python -m src.model_training.classification_training
```

for classification and, for regression:

```sh
python -m src.model_training.regression_training
```

In order to ensure the necessary artifacts are available for the API, one could also run the `API_start.sh` script, which creates the artifacts if needed and starts the API locally as well:

```sh
bash API_start.sh
```

Once the API is started locally, one can checkout the docs at `http://127.0.0.1:8000/docs#` for schemas and sample requests.

There are two routers, one for `regression/` and one for `classification/`. They basically contain the same endpoints.

### GET endpoints
- `/model_info/` giving information about the data and the type of model showcased.
  example requests:


**classification**
    ```bash
    curl -X 'GET' \
    'http://127.0.0.1:8000/classification/model_info/' \
    -H 'accept: application/json'
    ```

**regression**
    ```bash
    curl -X 'GET' \
    'http://127.0.0.1:8000/regression/model_info/' \
    -H 'accept: application/json'
    ```

- `/report/` summarizing model training metrics
  example requests:


**classification**
    ```bash
    curl -X 'GET' \
    'http://127.0.0.1:8000/classification/report/' \
    -H 'accept: application/json'
    ````

**regression**
    ```bash
    curl -X 'GET' \
    'http://127.0.0.1:8000/regression/report/' \
    -H 'accept: application/json'
    ```

### POST endpoints

- `/counterfactuals/` generating the counterfactuals for a base instance
  example requests:


**classification**
    ```bash
    curl -X 'POST' \
  'http://127.0.0.1:8000/classification/counterfactuals/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "instance": {
    "age": 0,
    "education_num": 0,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 0,
    "marital_status": "Married-civ-spouse",
    "relationship": "Husband",
    "workclass": "Private",
    "sex_Male": 0,
    "sex_Female": 0,
    "race": "White",
    "native_country": "United-States",
    "outcome": 0
  },
  "n_counterfactuals": 3,
  "method": "genetic",
  "fix_vars": [
    "string"
  ],
  "one_hot_encoded": [
    "string"
  ],
  "desired_outcome": 0}'
  ```

**regression**
  ```bash
  curl -X 'POST' \
  'http://127.0.0.1:8000/regression/counterfactuals/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "instance": {
    "MedInc": 0,
    "HouseAge": 0,
    "AveRooms": 0,
    "AveBedrms": 0,
    "Population": 0,
    "AveOccup": 0,
    "Latitude": 0,
    "Longitude": 0,
    "outcome": 0
  },
  "n_counterfactuals": 3,
  "method": "genetic",
  "fix_vars": [
    "string"
  ],
  "one_hot_encoded": [
    "string"
  ],
  "lower_limit": 0,
  "upper_limit": 0
}'```


## Project Structure

```text
.
|-api/ # Code to run the API
|-artifacts/ # Artifacts needed for the demos
|-notebooks/ # notebooks demonstrating how to generate counterfactuals
|-src/ # main code repository
|--cf_code/ # code to generate counterfactuals
|--data_loader/ # code to load the demo data from scikit-learn
|--model_training/ # code to train demo classification and regression models
|--pydantic_models/ # models for the API
|-API_start.sh # script to generate demo artifacts and start the API
|-pyproject.toml # file listing the necessary dependencies of the package
|-README.md # this file - describing the features of the package
|-uv.lock # project dependencies
```

## Disclaimer

The code in this repository was developed as a learning project. It is not intended to be used in any real application and it does not come with any guarantee that it will work every time.
Testing was done based on some examples and it works well based on those, though some generalization has been strived for.
I am aware of improvements that can be done. Specifically:
- the API isn't hosted anywhere, it is only a sample structure 
- unit testing is currently missing as well as some type hinting within the code
- logging could be added in order to report on what is going on under the hood and where the algorithms most spend their time on
- a user interface or at the very least, some plotting capability to allow visualization of the counterfactuals would be nice to have
- parametrizing mutation parameters (amount and probability of mutations) would be nice in order to better explore the solution space and strike a balance between convergence speed and solution diversity

## Further development

Some areas for further development are clear to me:
- a docker would be nice to have and will likely be added at some point
- adding further algorithms to generate counterfactuals could be interesting. I am thinking of adding simulated annealing at least, but some reinforcement learning approach would be cool; this depends entirely on the time I can find to keep working on this project
