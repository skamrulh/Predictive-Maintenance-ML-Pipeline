"""
Evaluation script for RUL prediction model performance.
Calculates MAE, R² score and other regression metrics.
"""

import pandas as pd
import argparse
import yaml
import joblib
import json
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, Any

# FILENAME CORRECTION: Import from common.py
from common import build_rul 


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml") -> None:
    """Main evaluation function for RUL model."""
    print("Evaluating RUL model performance...")
    
    # FILENAME CORRECTION: Use config.yaml
    cfg = load_config(config_path)
    
    # Load data
    st = pd.read_csv(cfg["paths"]["raw_stream"])
    feats = ["temp", "vibration", "pressure", "current", "sound"]
    
    # Load trained model
    try:
        pipe = joblib.load(cfg["paths"]["model_rul"])
        print("Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return
    
    # Build RUL dataset and prepare evaluation data
    df = build_rul(st, cap=cfg["rul"]["max_cycle_hours"])
    
    # Use recent data for evaluation (last 30% of machines)
    msk = (st["machine_id"] >= st["machine_id"].max() * 0.7)
    X = df.loc[msk, feats]
    y_true = df.loc[msk, "RUL_hours"]
    
    print(f"Evaluation data: {len(X)} samples")
    print(f"True RUL - Mean: {y_true.mean():.2f}, Std: {y_true.std():.2f}")
    
    # Generate predictions
    predictions = pipe.predict(X)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    # Prepare output
    output = {
        "mae": float(mae),
        "r2": float(r2),
        "evaluation_samples": len(X)
    }
    
    # Save metrics
    with open(cfg["paths"]["metrics_rul"], "w") as f:
        json.dump(output, f, indent=2)
    
    # Print results
    print(f"\n📊 RUL MODEL RESULTS:")
    print(f"MAE: {mae:.2f} hours")
    print(f"R² Score: {r2:.4f}")
    print(f"Metrics saved: {cfg['paths']['metrics_rul']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RUL model")
    # FILENAME CORRECTION: Use config.yaml
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    main(args.config)