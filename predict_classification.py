"""
Batch prediction script for classification model.
Generates failure probabilities for new data in batch mode.
"""

import pandas as pd
import argparse
import yaml
import joblib


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path, input_file, output_file):
    """Main prediction function."""
    print(f"Running batch prediction: {input_file} -> {output_file}")
    
    # Load configuration
    cfg = load_config(config_path)
    
    # Load trained model
    try:
        pipe = joblib.load(cfg["paths"]["model_cls"])
        print("Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return
    
    # Load input data
    df = pd.read_csv(input_file)
    print(f"Input data: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Prepare features (drop label and metadata columns if present)
    drop_cols = ["label_fail_within_24h", "machine_id", "hour"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Generate predictions
    probabilities = pipe.predict_proba(X)[:, 1]
    df["fail_within_24h_proba"] = probabilities
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"✅ Predictions saved: {output_file}")
    print(f"Prediction statistics:")
    print(f"  Mean probability: {probabilities.mean():.4f}")
    print(f"  Max probability: {probabilities.max():.4f}")
    print(f"  Samples > 0.5: {(probabilities > 0.5).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction for classification")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()
    main(args.config, args.input, args.output)