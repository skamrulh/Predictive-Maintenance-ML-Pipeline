"""
Training script for the failure classification model.
Loads tabular data, trains a RandomForestClassifier, and saves the model.
"""

import pandas as pd
import yaml
import joblib
import os
from common import pre_cls  # Import pipeline from common.py
from typing import List


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path: str = "config.yaml") -> None:
    """Main training function for classification model."""
    print("Training classification model...")
    
    cfg = load_config(config_path)
    paths = cfg["paths"]
    
    os.makedirs(paths["model_dir"], exist_ok=True)
    
    try:
        df = pd.read_csv(paths["tabular_train"])
    except FileNotFoundError:
        print(f"❌ Error: Training data not found at {paths['tabular_train']}")
        print("Please run 'python generate_synthetic_data.py' first.")
        return

    print(f"Loaded training data: {df.shape[0]} samples")

    # CRITICAL FIX: Define features to use only the 5 mean features
    # These 5 features will be used as the proxy for instantaneous readings in the API.
    target = "label_fail_within_24h"
    sensor_cols: List[str] = ["temp", "vibration", "pressure", "current", "sound"]
    features: List[str] = [f"{col}_mean" for col in sensor_cols]
    
    # Filter the training data to only include the new 5 features
    X_train = df[features]
    y_train = df[target]
    
    print(f"Training with {len(features)} features: {features}")
    
    # Create the model pipeline
    pipe = pre_cls(features)
    
    print("Fitting model...")
    pipe.fit(X_train, y_train)
    
    # Save the trained model
    model_path = paths["model_cls"]
    joblib.dump(pipe, model_path)
    
    print(f"✅ Classification model saved successfully: {model_path}")

if __name__ == "__main__":
    main()