# Predictive Maintenance Risk Pipeline

Train and serve a failure-risk model on the UCI AI4I 2020 dataset. The project includes leakage-safe preprocessing, an ordered 70/15/15 holdout, evaluation reports, a FastAPI scoring endpoint, and a feature-drift check.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py --output artifacts
uvicorn src.api:app --reload
python -m unittest discover -s tests -v
```

The first training run fetches dataset 601 through `ucimlrepo`; attribution: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), CC BY 4.0. UDI ordering is a proxy for chronology, not a timestamp. See [model card](docs/model_card.md).

`POST /predict` accepts sensor values named `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. `/health` reports artifact readiness. Outputs are educational risk estimates, not safety decisions.
