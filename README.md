# PowerForecastingProject

## Layout

```
data/        manifest.json, data_dictionary.md (raw files are not committed)
src/         build_daily.py, forecast_pipeline.py, evaluate.py, config.py
notebooks/   exploration only, outputs cleared before committing
results/     comparison_table.csv, run_metadata.json
docs/        worklogs, report drafts
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
```

Run the scripts as modules from the repo root, e.g. `python -m src.build_daily`.

## Rules

- `main` is protected. Every change goes through a pull request.
- The random seed lives in `src/config.py` (`SEED = 42`). Import it; never type `42` anywhere else.
- Clear notebook outputs before committing.
