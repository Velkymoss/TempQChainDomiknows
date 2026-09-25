# TempQChain



Fork from https://github.com/HLR/SpaRTUNQChain

Modified code from paper https://arxiv.org/abs/2406.13828 published by Premsri and Kordjamshidi used for temporal relationships between events.



## Setup

### Prerequisites
- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Dependencies
Install uv if not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or alternatively via pip:
```bash
pip install uv
```
Then in the project root use:
```bash
uv venv
source .venv/bin/activate
uv sync
```
### Data
We are using a dense version of the [TimeBank corpus](https://aclanthology.org/P14-2082/). 

Create a data folder in the root of the porject with following structure:
- [TimebankDense.full.txt](https://www.usna.edu/Users/cs/nchamber/caevo/TimebankDense.full.txt)
- timebank_1_2 (folder with annotated TB articles, not freely available unfornatunately)

Then run the create-data script to create the train/dev/test-split for TB-dense:
```bash
q-chain create-data
```


## CLI Usage

The CLI supports FR mode for open-ended questions and YN mode for Yes-No questions about temporal relationships.

## FR Mode
```bash
q-chain temporal-fr [OPTIONS]
```

### YN Mode
```bash
q-chain temporal-yn [OPTIONS]
```

## Experiment Tracking

This project uses MLflow for experiment tracking.  
Enable MLflow with the `--use-mlflow` flag:
```bash
q-chain temporal-fr --use-mlflow
```
MLflow will log metrics, hyperparameters and model artifacts 
under Temporal_FR/Temporal_YN.<br>
By default, logs are saved locally in mlruns/. 
You can browse them with:
```bash
mlflow ui
```

## Tests
### Unit Tests

```bash
uv run pytest
```
### Graph Tests
Graph tests must run in isolation due to side effects.
```bash
uv run python tests/graphs/run_tests.py
```

## Known Issues

### DomiKnows Library Bug Fix

Due to a bug in the `domiknows==0.533` library, you need to manually fix one line of code after installation:

**File:** `.venv/lib/python3.12/site-packages/domiknows/program/program.py` (line ~37)

**Change from:**
```python
if self.programName.index('.') >= 0:
```

**Change to:**
```python
if '.' in self.programName:
```

**Why this is needed:** The library uses `.index('.')` which throws a `ValueError` when no dot is found in the program name. This happens when using CLI entry points like `q-chain temporal-fr` where the name doesn't have a file extension.