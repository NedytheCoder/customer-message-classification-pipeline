from dataclasses import dataclass, field
from config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

@dataclass
class FeatureConfig:
    text_columns: list = field(default_factory=lambda: list(Config.TEXT_COLUMNS))
    max_features: int = 2000
    min_df: int = 2
    max_df: float = 0.90
    ngram_range: tuple = (1, 1)
    sublinear_tf: bool = True
    stop_words: str = "english"  # NOTE: English-only; multilingual caveat

class TfidfFeaturizer:
    def __init__(self, config=None):
        self.config = config or FeatureConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            ngram_range=self.config.ngram_range,
            sublinear_tf=self.config.sublinear_tf,
            stop_words=self.config.stop_words
        )
        self.fitted = False

    def combine_text(self, df):
        present = [c for c in self.config.text_columns if c in df.columns]
        if not present:
            raise KeyError(
                f"Turns out there ain't none of this {self.config.text_columns} in the df. The df only has {list(df.columns)})"
            )
        combined = df[present[0]].fillna("").astype(str)
        for col in present[1:]:
            combined = combined + " " + df[col].fillna("").astype(str)
        return combined.str.strip()

    def fit(self, df):
        text = self.combine_text(df)
        self.vectorizer.fit(text)
        self.fitted = True
        return self

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("You gotta fit first bro")
        return self.vectorizer.transform(self.combine_text(df))

    def get_feature_names(self):
        return list(self.vectorizer.get_feature_names_out())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
