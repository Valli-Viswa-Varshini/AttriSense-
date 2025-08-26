import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

try:
    import google.generativeai as genai  # optional
except Exception:
    genai = None

class DataAgent:
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

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
        """Basic cleaning + lightweight EDA.
        Returns: (df_clean, eda_report_text, plot_paths)
        """
        df = df.copy()

        # Strip column names
        df.columns = [str(c).strip() for c in df.columns]

        # Standardize common Yes/No
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'yes': 'Yes', 'no': 'No', 'TRUE': 'Yes', 'FALSE': 'No', 'True': 'Yes', 'False': 'No'})

        # Simple missing handling (do not impute here; pipeline will handle it)
        # Just report missing counts
        missing = df.isna().sum().to_dict()

        # EDA plots
        plots = []
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()

        # Numeric histograms
        for col in num_cols[:10]:  # cap to 10 to keep it lightweight
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), ax=ax, kde=False)
            ax.set_title(f"Distribution: {col}")
            fig_path = Path(f"eda_num_{col}.png")
            fig.savefig(fig_path, bbox_inches='tight')
            plots.append(str(fig_path))
            plt.close(fig)

        # Categorical bars
        for col in cat_cols[:10]:
            fig, ax = plt.subplots()
            df[col].value_counts(dropna=False).head(20).plot(kind='bar', ax=ax)
            ax.set_title(f"Top categories: {col}")
            fig_path = Path(f"eda_cat_{col}.png")
            fig.savefig(fig_path, bbox_inches='tight')
            plots.append(str(fig_path))
            plt.close(fig)

        # Basic text summary (optionally via LLM if available)
        eda_report = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "missing": missing,
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols
        }
        eda_text = f"EDA summary: {eda_report}"

        if self.llm_provider == "google" and genai is not None and os.getenv("GOOGLE_API_KEY"):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"Explain this HR dataset to a non-technical HR manager: {eda_report}"
                eda_text = model.generate_content(prompt).text
            except Exception:
                pass

        return df, eda_text, plots
