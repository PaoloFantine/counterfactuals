# counterfactuals

A Python toolkit for generating interpretable counterfactual explanations for both classification and regression models.
Built on top of scikit-learn compatible models and the explainerdashboard package, this project enables model interpretability and counterfactual prototype search as well as genetic algorithm-based explanations.

A demo of the functionality can be found in `notebooks/basic_demo.ipynb`

## Features

- Prototype-based counterfactual search using KDTree - looking for data instances close to the examined one in feature space that show the desired behavior

- Genetic algorithm counterfactual generation for actionable, realistic changes - Nondominated Sorting Genetic Algorithm II (NSGA-II) with multi-objective optimization [https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf](https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf), [https://arxiv.org/pdf/2004.11165](https://arxiv.org/pdf/2004.11165)

- Support for both (multiclass) classification and regression tasks

- Utility scripts for loading sample datasets, training models, and saving sample artifacts

- Local API to showcase an example on how to serve counterfactuals interactively

## Getting started

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

## Usage

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


## Project Structure

```text
.
|-api/ # Code to run the API
|-artifacts/ # Artifacts needed for the demos
|-notebooks/ # notebooks demonstrating how to generate counterfactuals
|-src/ # main code repository
|--cf_code/ # code to generate counterfactuals
|--data_loader/ # code to load the demo data from scikit-learn
|--model_training_scripts/ # code to train demo classification and regression models
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
- unit testing is currently missing as well as some type hinting within the code
- logging could be added in order to report on what is going on under the hood and where the algorithms most spend their time on
- a user interface or at the very least, some plotting capability to allow visualization of the counterfactuals would be nice to have
- parametrizing mutation parameters (amount and probability of mutations) would be nice in order to better explore the solution space and strike a balance between convergence speed and solution diversity

## Further development

Some areas for further development are clear to me:
- a docker would be nice to have and will likely be added at some point
- adding further algorithms to generate counterfactuals could be interesting. I am thinking of adding simulated annealing at least, but some reinforcement learning approach would be cool; this depends entirely on the time I can find to keep working on this project
