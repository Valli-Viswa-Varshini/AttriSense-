import argparse
import pandas as pd
from agents.prediction_agent import PredictionAgent

parser = argparse.ArgumentParser(description="Train the attrition model")
parser.add_argument("--csv", default="data/sample_hr_with_attrition.csv",
                    help="Path to training CSV with 'Attrition' target column")
parser.add_argument("--model_path", default="models/attrition_model.pkl",
                    help="Where to save the trained model")
args = parser.parse_args()

print(f"Loading data from {args.csv}")
df = pd.read_csv(args.csv)

agent = PredictionAgent(model_path=args.model_path)
res = agent.train(df, target_column="Attrition")
print("Training complete.")
print("Metrics:", res.metrics)
print("Saved:", res.model_path)
