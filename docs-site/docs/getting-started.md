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

3. Check out the demo at `./notebooks/basic_demo.ipynb` or a more high-level overview in the docs