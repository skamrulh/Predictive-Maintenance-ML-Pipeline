"""
Training script for the Remaining Useful Life (RUL) regression model.
Loads raw data, builds RUL labels, trains a GradientBoostingRegressor.
"""

import pandas as pd
import yaml
import joblib
import os
# Import pipeline and RUL builder from common.py
from common import pre_rul, build_rul

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path="config.yaml"):
    """Main training function for RUL model."""
    print("Training RUL model...")
    
    # Load configuration
    cfg = load_config(config_path)
    paths = cfg["paths"]
    
    # Create models directory if it doesn't exist
    os.makedirs(paths["model_dir"], exist_ok=True)
    
    # Load raw sensor stream data
    try:
        stream = pd.read_csv(paths["raw_stream"])
    except FileNotFoundError:
        print(f"❌ Error: Raw stream data not found at {paths['raw_stream']}")
        print("Please run 'python generate_synthetic_data.py' first.")
        return

    print(f"Loaded raw stream data: {len(stream)} records")

    # Build the RUL dataset
    print("Building RUL labels...")
    df = build_rul(stream, cap=cfg["rul"]["max_cycle_hours"])
    
    # Define features and target
    # These are the raw sensor values, as defined in evaluate_rul.py
    features = ["temp", "vibration", "pressure", "current", "sound"]
    target = "RUL_hours"
    
    X_train = df[features]
    y_train = df[target]
    
    print(f"Training with {len(features)} features.")
    
    # Create the model pipeline
    pipe = pre_rul(features)
    
    # Train the model
    print("Fitting model...")
    pipe.fit(X_train, y_train)
    
    # Save the trained model
    model_path = paths["model_rul"]
    joblib.dump(pipe, model_path)
    
    print(f"✅ RUL model saved successfully: {model_path}")

if __name__ == "__main__":
    main()