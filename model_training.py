from pathlib import Path
from config import Config
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from process import preprocess_training
from load_data import get_final_dataframe
from sklearn.model_selection import train_test_split
from features import TfidfFeaturizer
import joblib
from datetime import datetime
import json

TARGETS = [Config.CLASS_COL] + Config.SUBLABELS

def build_models():
    return {
        "random_forest": MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=300,
                random_state=Config.SEED,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
        ),
        "logistic_regression": MultiOutputClassifier(
            LogisticRegression(max_iter=1000, random_state=Config.SEED)
        ),
    }
    
def train_and_store(artifact_dir: Path = Config.ARTIFACT_DIR):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    df = preprocess_training(get_final_dataframe())
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=Config.SEED, stratify=df[Config.CLASS_COL]
    )
 
    featurizer = TfidfFeaturizer()
    X_train = featurizer.fit(train_df).transform(train_df)
    Y_train = train_df[TARGETS].astype(str)

    models = build_models()
    for name, model in models.items():
        model.fit(X_train, Y_train)
        joblib.dump(model, artifact_dir / f"model_{name}.joblib")

    featurizer.save(artifact_dir / "vectorizer.joblib")
    joblib.dump(test_df, artifact_dir / "test_data.joblib")

    data_on_model_and_features = { 
        "model_version": datetime.now().strftime("%Y%m%d_%H%M%S"), 
        "targets": TARGETS,
        "models": list(models.keys()),
        "n_features": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(len(test_df)),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(data_on_model_and_features, indent=2))
    return data_on_model_and_features

if __name__ == "__main__":
    trained = train_and_store()
    print("\n=== training complete ===")
    print(json.dumps(trained, indent=2))
