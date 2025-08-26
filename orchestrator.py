from typing import Dict

import pandas as pd

from agents.data_agent import DataAgent
from agents.prediction_agent import PredictionAgent
from agents.analysis_agent import AnalysisAgent
from agents.explanation_agent import ExplanationAgent

class Orchestrator:
    """
    Single entry point that triggers the right agents based on the dataset:
    - If target column exists -> training
    - If not -> prediction
    """
    def __init__(self, model_path: str, llm_provider: str = "google", target_column: str = "Attrition"):
        self.target_column = target_column
        self.data_agent = DataAgent(llm_provider=llm_provider)
        self.prediction_agent = PredictionAgent(model_path)
        self.analysis_agent = AnalysisAgent(llm_provider=llm_provider)
        self.explanation_agent = ExplanationAgent(llm_provider=llm_provider)

    def run(self, df: pd.DataFrame) -> Dict:
        has_target = self.target_column in df.columns
        if has_target:
            return self.run_train(df)
        else:
            return self.run_predict(df)

    def run_train(self, df: pd.DataFrame) -> Dict:
        processed_df, eda_report, plots = self.data_agent.process(df)
        result = self.prediction_agent.train(processed_df, target_column=self.target_column)
        out = {
            "mode": "train",
            "eda_report": eda_report,
            "eda_plots": plots,
            "train_metrics": result.metrics,
            "model_path": result.model_path,
        }
        return out

    def run_predict(self, df: pd.DataFrame) -> Dict:
        processed_df, eda_report, plots = self.data_agent.process(df)
        predictions = self.prediction_agent.predict(processed_df)
        analysis, analysis_plots = self.analysis_agent.analyze(processed_df, predictions)
        return {
            "mode": "predict",
            "eda_report": eda_report,
            "eda_plots": plots,
            "predictions": predictions,
            "analysis": analysis,
            "analysis_plots": analysis_plots
        }
