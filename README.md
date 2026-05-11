# Experiments with MLflow

This repository is a hands-on MLflow learning guide that covers:

- Local experiment tracking on your machine
- Remote tracking on DagsHub
- Manual logging vs `mlflow.autolog()`
- Hyperparameter tuning with nested runs
- Artifact, dataset, model, and tag logging
- Model Registry concepts and workflow

The goal is to help you understand the full experiment lifecycle, from single training runs to reproducible, shareable MLOps workflows.

## What This Repository Demonstrates

1. Manual logging of metrics, params, artifacts, and models.
2. Automatic logging using `mlflow.autolog()`.
3. Local MLflow tracking with a SQLite backend.
4. Remote MLflow tracking to DagsHub.
5. Hyperparameter tuning with `GridSearchCV` and nested MLflow runs.
6. Input dataset snapshot logging via `mlflow.data`.
7. Model Registry lifecycle concepts (development, staging, production, archive).

## Repository Structure

| Path | Purpose |
|---|---|
| `src/file1.py` | Local tracking example with manual logging (wine classification). |
| `src/autolog.py` | Local tracking example using `mlflow.autolog()`. |
| `src/hypertune_local.py` | Local hyperparameter tuning with nested runs (breast cancer dataset). |
| `src/file2.py` | Remote tracking to DagsHub with manual logging. |
| `src/hypertune_remote.py` | Remote hyperparameter tuning to DagsHub with nested runs. |
| `ml-flow-basics.txt` | Notes on what MLflow can log. |
| `mlflow-autolog.txt` | Notes on `mlflow.autolog()` coverage and limitations. |
| `model-registry.txt` | Notes on model registry lifecycle and governance. |
| `mlflow.db` | Local SQLite backend store for run metadata. |
| `mlartifacts/` | Local artifact storage used by tracking server workflows. |
| `mlruns/` | Local run/artifact directories created by MLflow. |

## Prerequisites

- Python 3.9+ (recommended)
- `pip`
- (Optional, for remote tracking) DagsHub account and access token

## Setup

Run from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install mlflow scikit-learn pandas matplotlib seaborn dagshub
```

## Local MLflow Workflow

### 1) Start the local MLflow tracking server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5050
```

Open UI:

`http://127.0.0.1:5050`

### 2) Run local experiment scripts

```bash
python3 src/file1.py
python3 src/autolog.py
python3 src/hypertune_local.py
```

What these scripts log locally:

- Experiment names: `MLOPS-1`, `breast-cancer-rf-hp`
- Metrics and parameters
- Source script as artifact
- Trained model artifact
- Confusion matrix plot (`Confusion-matrix.png`) in the wine examples
- Nested child runs in hyperparameter tuning
- Training/testing data snapshots in tuning script

## Remote MLflow Workflow (DagsHub)

The remote scripts are already configured for this tracking URI:

`https://dagshub.com/dhanushpavann/Experiments-with-MLFlow-.mlflow`

### 1) Authenticate with DagsHub

You can use any one of these methods:

- `dagshub login`
- `dagshub login --token <your_token>`
- `export DAGSHUB_USER_TOKEN=<your_token>`

For explicit MLflow basic auth (especially useful in CI):

```bash
export MLFLOW_TRACKING_USERNAME=<dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<dagshub_token>
```

### 2) Run remote experiment scripts

```bash
python3 src/file2.py
python3 src/hypertune_remote.py
```

### 3) Open DagsHub MLflow UI

`https://dagshub.com/dhanushpavann/Experiments-with-MLFlow-.mlflow`

You can inspect runs, compare them, view artifacts, and register models from the same UI.

## Script-by-Script Tracking Behavior

| Script | Tracking Target | Key Behavior |
|---|---|---|
| `src/file1.py` | `http://127.0.0.1:5050` | Manual logging of params, metric, tags, confusion matrix artifact, model. |
| `src/autolog.py` | `http://127.0.0.1:5050` | Uses `mlflow.autolog()` for automatic model/param/metric logging. |
| `src/hypertune_local.py` | `sqlite:///mlflow.db` | `GridSearchCV` + nested runs; logs dataset snapshots and best model locally. |
| `src/file2.py` | DagsHub MLflow URL | Same as `file1.py`, but logs remotely. |
| `src/hypertune_remote.py` | DagsHub MLflow URL | Same as local tuning script, but logs remotely to DagsHub. |

## Model Registry Notes

- This repo logs models with `mlflow.sklearn.log_model(...)`.
- From either local UI or DagsHub UI, you can register logged models and version them.
- Typical lifecycle stages discussed in this guide: `Development/None`, `Staging`, `Production`, `Archived`.

## Troubleshooting

`Connection refused` on local scripts:
- Ensure the local server is running on `127.0.0.1:5050`.
- Verify no other process is using port `5050`.

Runs not appearing where expected:
- Check each script’s `mlflow.set_tracking_uri(...)` value.
- Local and remote scripts intentionally point to different URIs.

Remote auth errors (`401`, `403`):
- Re-login with `dagshub login`.
- Re-check `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`.
- Confirm you have contributor access to the DagsHub repo.

Model Registry tab issues in local UI:
- Use a database backend (`sqlite:///mlflow.db`) as shown above.

## Suggested Learning Order

1. Run `src/file1.py` to understand manual logging.
2. Run `src/autolog.py` to compare manual vs automatic logging.
3. Run `src/hypertune_local.py` to learn nested runs and dataset logging.
4. Run `src/file2.py` and `src/hypertune_remote.py` to publish runs to DagsHub.
5. Register one model in MLflow UI and move it through lifecycle stages.

## License

This repo is licensed under the terms of the [LICENSE](LICENSE) file.
