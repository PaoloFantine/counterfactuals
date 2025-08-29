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
