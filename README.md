# Customer Interaction Classification — Multi-Label ML Pipeline

A modular, multi-label machine-learning system that classifies customer support
messages across three label dimensions (`y2`, `y3`, `y4`). It refactors the
original single-label prototype into clearly separated stages: data loading,
preprocessing, feature extraction, model training, batch inference, and
evaluation.

## Project structure

| File | Responsibility |
|------|----------------|
| `config.py` | Central configuration (column names, targets, paths, random seed) |
| `load_data.py` | Load, combine and validate the raw CSV files |
| `process.py` | Preprocessing: shared text normalisation (train + inference) and training-only steps (de-duplication, label cleaning, filtering) |
| `features.py` | Leakage-free TF-IDF featuriser (fit on train only; saveable/loadable) |
| `model_training.py` | Train the multi-label models and store all artefacts |
| `predict.py` | Batch inference on new messages using the saved artefacts |
| `evaluation.py` | Evaluate the models and produce metrics, reports and confusion matrices |

Artefacts are written to `artifacts/`; evaluation outputs and predictions to `outputs/`.

## Install dependencies

```bash
pip install -r requirements.txt
```

## 1. Train the models (produces artefacts)

Runs the full training pipeline (load → preprocess → split → TF-IDF fit on train
only → train Random Forest and Logistic Regression) and saves the vectoriser,
both models, the held-out test set and a metadata file to `artifacts/`.

```bash
python model_training.py
```

## 2. Run batch prediction on new messages

Loads the saved artefacts and classifies a CSV of new messages, writing a
predictions file with the predicted labels, confidence scores, model name/version
and a timestamp.

```bash
python predict.py --input data/new_messages.csv --output outputs/predictions.csv
```

Optional: choose which trained model to use (default `random_forest`):

```bash
python predict.py --model logistic_regression
```

## 3. Evaluate the models

Loads the saved artefacts, evaluates both models on the held-out test set across
all three labels, and writes metrics, per-label classification reports and
confusion matrices to `outputs/` (also runs a batch prediction with the best model).

```bash
python evaluation.py
```

## Outputs

- `artifacts/` — `vectorizer.joblib`, `model_*.joblib`, `test_data.joblib`, `metadata.json`
- `outputs/` — `metrics.json`, `classification_report_*.txt`, `confusion_matrix_*_y2.png`, `predictions.csv`
