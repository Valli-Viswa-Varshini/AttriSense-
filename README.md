# AttriSense – Agentic HR Attrition Predictor (Fixed)

This is a **fixed, production-ready draft** of your agentic HR attrition project with:

- ✅ A **reliable training pipeline** that expects a dataset **with** `Attrition` (target) + features.
- ✅ A **prediction flow** for HR uploads **without** the `Attrition` column.
- ✅ A robust **orchestrator** that triggers the right agents automatically.
- ✅ A safer `.env.example` (no secrets committed) and pinned `requirements.txt` to avoid pickle errors.
- ✅ Clear sample data and a one-command training script.

---

## Quickstart

```bash
# 1) Create a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure environment
cp .env.example .env
# put your API keys only into .env; do NOT commit .env to git

# 4) Train the model (uses data/sample_hr_with_attrition.csv by default)
python train.py

# 5) Run the app
streamlit run app/app.py
```

---

## How it works

- If you **upload data with `Attrition`**, the app lets you **train** and evaluates the model (accuracy, precision, recall, F1).
- If you **upload data without `Attrition`**, the app switches to **predict** mode and returns an `Attrition_Prediction` column.
- The **Orchestrator** decides which agents to trigger:
  - `DataAgent` → light EDA + cleaning
  - `PredictionAgent` → train or predict (via scikit-learn Pipeline)
  - `AnalysisAgent` → summarize predictions
  - `ExplanationAgent` → optional LLM explanation (Gemini), gracefully falls back if API key is missing

---

## Files

- `train.py` – CLI training script
- `agents/` – all agents
- `app/app.py` – Streamlit UI
- `data/sample_hr_with_attrition.csv` – sample training data
- `models/` – stores `attrition_model.pkl`
- `utils/metrics.py` – metric helpers
- `config.yaml` – simple config (e.g., target column, model path)

---

## Why the original pickle failed

Your previous `attrition_model.pkl` was created with **NumPy 2.x** (which uses `numpy._core`) but was being loaded in an environment with **NumPy 1.x**, causing:

```
ModuleNotFoundError: No module named 'numpy._core'
```

**Fix:** This project **pins versions** and provides a **retrainable** pipeline. Always rebuild the pickle in the same or compatible environment, or keep versions pinned in `requirements.txt`.

---

## Notes
- Keep `.env` local and *never* commit your real API keys.
- If you change features or encoding, retrain the model (`python train.py`) to update `models/attrition_model.pkl`.
