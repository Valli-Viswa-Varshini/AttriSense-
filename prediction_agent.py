import os
from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils.metrics import classification_metrics

@dataclass
class TrainResult:
    metrics: dict
    model_path: str

class PredictionAgent:
    def __init__(self, model_path: str = "models/attrition_model.pkl"):
        self.model_path = model_path
        self.model: Optional[Pipeline] = None
        # Try to load an existing model
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                # Fail gracefully; user can retrain
                print(f"[WARN] Could not load model from {self.model_path}: {e}")

    # ---------- helpers ----------
    def _detect_columns(self, df: pd.DataFrame) -> Tuple[list, list]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        return num_cols, cat_cols

    def _target_to_numeric(self, y_series: pd.Series) -> pd.Series:
        # Map common labels to 1 (Yes/Leaver) and 0 (No/Stay)
        mapping = {
            "yes": 1, "y": 1, "true": 1, "1": 1, 1: 1,
            "no": 0, "n": 0, "false": 0, "0": 0, 0: 0
        }
        return y_series.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)

    # ---------- training ----------
    def train(self, df: pd.DataFrame, target_column: str = "Attrition") -> TrainResult:
        assert target_column in df.columns, f"Training requires '{target_column}' in columns"
        y = self._target_to_numeric(df[target_column])
        X = df.drop(columns=[target_column])

        num_cols, cat_cols = self._detect_columns(X)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols),
            ]
        )

        clf = LogisticRegression(max_iter=1000)
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = classification_metrics(y_test, y_pred)

        # Persist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(pipe, self.model_path)
        self.model = pipe

        return TrainResult(metrics=metrics, model_path=self.model_path)

    # ---------- inference ----------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            # Try to load
            if os.path.exists(self.model_path):
                try:
                    self.model = joblib.load(self.model_path)
                except Exception as e:
                    raise RuntimeError(f"Model not available and could not be loaded: {e}")
            else:
                raise RuntimeError("Model not available. Train first to create the pickle.")

        df_out = df.copy()
        try:
            preds = self.model.predict(df_out)
        except Exception as e:
            # If the model expects certain columns, try selecting intersection
            model_features = None
            try:
                model_features = self.model.named_steps["preprocess"].get_feature_names_out()
            except Exception:
                pass
            raise RuntimeError(f"Prediction failed. Ensure input features match training schema. Error: {e}")

        df_out["Attrition_Prediction"] = ["Yes" if int(p) == 1 else "No" for p in preds]
        return df_out
