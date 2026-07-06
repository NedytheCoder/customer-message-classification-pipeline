from pathlib import Path
import json
from features import TfidfFeaturizer
import joblib
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, precision_recall_fscore_support)
import pandas as pd
from process import preprocess_inference
import matplotlib.pyplot as plt
from config import Config
from datetime import datetime

ARTIFACT_DIR = Config.ARTIFACT_DIR
OUTPUT_DIR = Config.OUTPUT_DIR
NEW_MESSAGES = Config.NEW_MESSAGES

def load_artifacts(artifact_dir: Path = ARTIFACT_DIR):
    meta_path = artifact_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run model_training.py first to create artifacts."
        )
    meta = json.loads(meta_path.read_text())
    featurizer = TfidfFeaturizer.load(artifact_dir / "vectorizer.joblib")
    test_df = joblib.load(artifact_dir / "test_data.joblib")
    models = {
        name: joblib.load(artifact_dir / f"model_{name}.joblib")for name in meta["models"]
    }
    return meta, featurizer, test_df, models

def compute_metrics(predictions, test_df, targets):
    result = {}
    for i, target in enumerate(targets):
        y_true = test_df[target].astype(str).to_numpy()
        y_pred = predictions[:, i].astype(str)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        result[target] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "per_class": classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            ),
        }
    return result

def report_text(name, predictions, test_df, targets):
    lines = [f"===== {name} =====", ""]
    for i, target in enumerate(targets):
        y_true = test_df[target].astype(str).to_numpy()
        y_pred = predictions[:, i].astype(str)
        lines.append(f"--- target: {target} ---")
        lines.append(classification_report(y_true, y_pred, zero_division=0))
    return "\n".join(lines)


def save_confusion_matrix(name, predictions, test_df, target, targets, out_path):
    i = targets.index(target)
    y_true = test_df[target].astype(str).to_numpy()
    y_pred = predictions[:, i].astype(str)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"{name} — {target}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def avg_macro_f1(model_metrics, targets):
    return sum(model_metrics[target]["f1_macro"] for target in targets) / len(targets)


def batch_prediction(featurizer, model, model_name, meta, targets, input_csv, out_path):
    df = pd.read_csv(input_csv, skipinitialspace=True)

    X = featurizer.transform(preprocess_inference(df))
    prediction = model.predict(X)              
    probability = model.predict_proba(X)          

    output = pd.DataFrame()
    output["message_id"] = (
        df["message_id"] if "message_id" in df.columns else range(1, len(df) + 1)
    )
    output["ticket_summary"] = df.get(Config.TICKET_SUMMARY, "")
    output["interaction_content"] = df.get(Config.INTERACTION_CONTENT, "")
    for i, target in enumerate(targets):
        output[f"pred_{target}"] = prediction[:, i]
        output[f"confidence_{target}"] = probability[i].max(axis=1).round(4)
    output["model_name"] = model_name
    output["model_version"] = meta["model_version"]
    output["prediction_timestamp"] = datetime.now().isoformat(timespec="seconds")

    output.to_csv(out_path, index=False)
    return output

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, featurizer, test_df, models = load_artifacts()
    targets = meta["targets"]

    # transform the held-out test set once with the saved vectorizer
    X_test = featurizer.transform(test_df)

    all_metrics: dict[str, dict] = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        all_metrics[name] = compute_metrics(preds, test_df, targets)

        (OUTPUT_DIR / f"classification_report_{name}.txt").write_text(
            report_text(name, preds, test_df, targets)
        )
        save_confusion_matrix(
            name, preds, test_df, Config.CLASS_COL, targets,
            OUTPUT_DIR / f"confusion_matrix_{name}_{Config.CLASS_COL}.png",
        )

    # persist all metrics (default=float converts numpy scalars)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, default=float)
    )

    # print a compact comparison table
    print("\n=== model comparison (test set) ===")
    header = f"{'model':22} {'label':6} {'acc':>6} {'f1_macro':>9} {'f1_weighted':>12}"
    print(header)
    print("-" * len(header))
    for name in models:
        for t in targets:
            m = all_metrics[name][t]
            print(f"{name:22} {t:6} {m['accuracy']:6.3f} "
                  f"{m['f1_macro']:9.3f} {m['f1_weighted']:12.3f}")

    # choose best model by average macro-F1 across the three labels
    best = max(models, key=lambda n: avg_macro_f1(all_metrics[n], targets))
    print(f"\nbest model (avg macro-F1): {best}")

    # batch inference on new messages with the best model's saved artifacts
    preds_df = batch_prediction(
        featurizer, models[best], best, meta, targets,
        NEW_MESSAGES, OUTPUT_DIR / "predictions.csv",
    )
    print("\n=== batch predictions on new_messages.csv ===")
    cols = ["message_id"] + [f"pred_{t}" for t in targets] + [f"confidence_{t}" for t in targets]
    print(preds_df[cols].to_string(index=False))
    print(f"\noutputs written to {OUTPUT_DIR}/")
    
if __name__ == "__main__":
    main()