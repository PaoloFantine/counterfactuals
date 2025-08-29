# counterfactuals

> Counterfactual explanations made simple: prototype search & genetic algorithm-based methods for interpretable ML.

A Python toolkit for generating **interpretable counterfactual explanations** for both classification and regression models.  
Built on top of scikit-learn compatible models, it enables:
- Prototype-based counterfactual search
- Genetic algorithm–based counterfactuals (NSGA-II, multi-objective optimization)
- Support for classification (including multiclass) and regression
- A demo API to serve counterfactuals interactively

📓 See the demo notebook: [`notebooks/basic_demo.ipynb`](./notebooks/basic_demo.ipynb)

---

## Features

- 🔍 **Prototype-based search** (KDTree) — find real data points similar to the target instance but with the desired outcome  
- 🧬 **Genetic counterfactuals** — generate realistic, actionable changes using NSGA-II  
- 📊 **Classification & regression support**  
- 🛠️ **Utility scripts** for dataset loading and model training  
- 🌐 **Local API** to explore counterfactuals interactively  

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone repository
git clone git@github.com:PaoloFantine/counterfactuals.git
cd counterfactuals

# Install uv (Linux/macOS example)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create & activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv lock
uv sync --locked --active
```

# Usage

## Try the notebook
```sh
jupyter notebook notebooks/basic_demo.ipynb
```

## Run the API

Ensure demo artifacts exist:

```sh
# For classification
python -m src.model_training.classification_training
```

```sh
# For regression
python -m src.model_training.regression_training
```

then start the API locally:
```sh
uv run uvicorn api.main:app --reload
```

or simply run the helper script that ensures the artifacts exist, creates them if they don't and starts the API locally:

```sh
bash API_start.sh
```

Once the API is running, visit http://127.0.0.1:8000/docs for auto-generated API docs.

There are two routers in the API:

- `/regression/` for regression models 
- `/classification/` for classification models

| **Endpoint**      | **Method**  | **Description**                               |
| ----------------- | ----------- | --------------------------------------------- |
| `/model_info/`    | GET         | Info about the model and dataset              |
| `/report/`        | GET         | Training metrics summary                      |
| `/counterfactuals`|POST         | Generate counterfactuals for a given instance |

See the [swagger docs](http://127.0.0.1:8000/docs) for full request/response schemas.

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

## Limitations and Roadmap
This project is a learning project, not production-ready. Current limitations:

- API is local only (no deployment setup yet)

- No unit tests or logging

- Limited configurability of genetic algorithm parameters

- No visualization or UI for counterfactuals

## Future directions:

- Docker support for easier deployment

- Additional counterfactual generation methods (e.g., simulated annealing, reinforcement learning)

- Improved logging, testing, and visualization

## License

MIT License — see LICENSE for details.
