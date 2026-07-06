"""Batch inference entry point (Task 3).

Loads the saved artefacts and classifies new customer messages. The actual
prediction is delegated to evaluation.batch_prediction, so training-time and
serving-time use exactly one implementation (training-serving consistency).

Usage:
    python predict.py --input data/new_messages.csv --output outputs/predictions.csv
    python predict.py --model logistic_regression
"""

import argparse

from config import Config
from evaluation import batch_prediction, load_artifacts


def run(input_csv, output_csv, model_name=None):
    """Classify the messages in input_csv and write predictions to output_csv."""
    meta, featurizer, _test_df, models = load_artifacts()
    targets = meta["targets"]

    # default to the first trained model if none requested
    if model_name is None:
        model_name = meta["models"][0]
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(models)}")

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return batch_prediction(
        featurizer, models[model_name], model_name, meta, targets, input_csv, output_csv
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-classify new customer messages.")
    parser.add_argument("--input", default=str(Config.NEW_MESSAGES),
                        help="input CSV of new messages")
    parser.add_argument("--output", default=str(Config.OUTPUT_DIR / "predictions.csv"),
                        help="where to write the predictions CSV")
    parser.add_argument("--model", default="random_forest",
                        help="which trained model to use (default: random_forest)")
    args = parser.parse_args()

    result = run(args.input, args.output, args.model)
    print(f"Wrote {len(result)} predictions to {args.output} using model '{args.model}'")
