# ML Creator

ML Creator is a bounded, source-linked research agent for discovering machine-learning features and model architectures before a dataset is available.

It uses iterative planning, web search, extraction, critique, and a separate admission review. It can search Kaggle, papers, and repositories. Structure review focuses on architecture and novelty, including hybrids such as XGBoost + LSTM; validation notes are stored only as context.

## Install

```bash
python3 -m pip install -e .
cp .env.example .env
```

Set the provider and search settings in `.env` (`JUSTWOKER_API_KEY`, `JUSTWOKER_BASE_URL`, `JUSTWOKER_MODEL`, and `SEARXNG_URL`).

## Run

```bash
ml-creator --config run.json --mode STRUCTURES \
  --requested-model-structures 3 --sources kaggle,papers,repos \
  --output structures.json
```

Other modes are `FEATURES` and `BOTH`:

```bash
ml-creator --config run.json --mode FEATURES --requested-features 5 --output features.json
ml-creator --config run.json --mode BOTH --requested-features 5 --output blueprint.json
```

Use `--problem` and `--horizon-days` for a specific prediction task. Omit the horizon, set it to `null`, or pass `--no-horizon` when it does not apply.

The same source selection can be stored as `research.sources` in `run.json`. Allowed values are `kaggle`, `papers`, and `repos`.

## Scope and limits

The workflow is `NO_DATA` research. Results are proposals supported by returned search snippets, not measured feature lift or model performance. The CLI rejects data-backed evaluation until a dataset adapter exists. Generated outputs and checkpoints are ignored by Git; checkpoints are audit snapshots and are not resumable.

Structure admission uses a separate reviewer call. Configure an independent reviewer with `REVIEW_BASE_URL`, `REVIEW_MODEL`, `REVIEW_API_PROTOCOL`, and `REVIEW_API_KEY` when available; otherwise the configured JustWoker endpoint is used for the reviewer prompt.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

`feature_scout.py` is a legacy standalone feature-research script and is not used by the `ml-creator` CLI.
