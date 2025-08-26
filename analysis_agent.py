import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except Exception:
    genai = None

class AnalysisAgent:
    def __init__(self, llm_provider: str = "google"):
        load_dotenv()
        self.llm_provider = llm_provider
        self._configure_llm()

    def _configure_llm(self):
        if self.llm_provider == "google" and genai is not None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                except Exception:
                    pass

    def analyze(self, df: pd.DataFrame, predictions_df: pd.DataFrame) -> Tuple[str, List[str]]:
        summary = predictions_df["Attrition_Prediction"].value_counts().to_dict()

        # Plot prediction counts
        fig, ax = plt.subplots()
        sns.countplot(x="Attrition_Prediction", data=predictions_df, ax=ax)
        ax.set_title("Attrition Predictions Count")
        fig_path = Path("analysis_plot.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        analysis_text = f"Predictions summary: {summary}"
        if self.llm_provider == "google" and genai is not None and os.getenv("GOOGLE_API_KEY"):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"Explain this attrition prediction summary to an HR manager in simple terms: {summary}"
                analysis_text = model.generate_content(prompt).text
            except Exception:
                pass

        return analysis_text, [str(fig_path)]
