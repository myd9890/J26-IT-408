# Severity-Based Resource Optimization

This student project is an AI-assisted decision-support system for Cavendish
banana plantations. It accepts disease predictions for plantation zones,
calculates an intervention priority, and uses Mixed-Integer Linear Programming
(MILP) to choose the best treatment plan under chemical, labour, budget, and
visit limits.

The project does not train an image-classification model. The `confidence`
value is designed to receive output from a separate disease-detection model.
This component focuses on prioritization and resource allocation.

## How it works

Each zone receives a priority score:

```text
priority = 0.40(severity) + 0.30(risk)
         + 0.10(confidence x 100) + 0.20(urgency)
```

The PuLP/CBC solver then maximizes the total priority of treated zones without
exceeding the available resources.

## Project structure

```text
data/                 Sample zone and resource inputs
results/              Reproducible scenario and baseline CSV results
src/api.py             FastAPI routes
src/main.py            Command-line demonstration and result generator
src/schemas/           Request validation models
src/services/          Scoring and optimization logic
tests/                 API tests
```

## Setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the sample

```powershell
python -m src.main
```

This prints the selected treatment plan and updates the CSV files in
`results/`.

## Run the API

```powershell
python -m uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to use the interactive API documentation.
The available endpoints are:

- `POST /api/v1/optimize` - generate the best allocation plan
- `POST /api/v1/scenarios` - compare normal and shortage scenarios
- `POST /api/v1/baselines` - compare MILP against severity-first selection

## Test

```powershell
python -m pytest -q
```

## Dataset note

The included records are a small simulated dataset for demonstrating the
optimization workflow. Results should not be used for real agricultural
treatment decisions without validated field data and expert review.
