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