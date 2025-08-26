import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from agents.orchestrator import Orchestrator

st.set_page_config(page_title="AttriSense Agentic AI", layout="wide")
st.title("AttriSense – Agentic HR Attrition Predictor")

st.markdown("""
**How to use**
- If your CSV has an `Attrition` column → click **Train Model**.
- If your CSV does not have `Attrition` → click **Predict Attrition**.
""")

uploaded = st.file_uploader("Upload Employee CSV", type=["csv"])
orch = Orchestrator(model_path="models/attrition_model.pkl")

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📄 Preview")
    st.dataframe(df.head())

    has_target = "Attrition" in df.columns

    st.subheader("🔍 EDA")
    processed_df, eda_text, plots = orch.data_agent.process(df)
    st.write(eda_text)
    cols = st.columns(3)
    for i, p in enumerate(plots[:6]):
        cols[i % 3].image(p, use_column_width=True)

    if has_target:
        st.success("Detected `Attrition` in your data. Ready to TRAIN.")
        if st.button("Train Model"):
            out = orch.run_train(df)
            st.subheader("✅ Training Complete")
            st.json(out["train_metrics"])
            st.info(f"Model saved to: {out['model_path']}")
    else:
        st.info("No `Attrition` column found. Ready to PREDICT.")
        if st.button("Predict Attrition"):
            out = orch.run_predict(df)
            st.subheader("🧮 Predictions (first 20 rows)")
            st.dataframe(out["predictions"].head(20))

            st.subheader("📈 Analysis")
            st.write(out["analysis"])
            for p in out["analysis_plots"]:
                st.image(p)

            st.subheader("💬 Ask Explanation Agent")
            query = st.text_input("Ask a question about predictions")
            if st.button("Ask"):
                answer = orch.explanation_agent.chat(query, {"summary": out["analysis"]})
                st.success(answer)
else:
    st.info("Upload a CSV to begin.")
