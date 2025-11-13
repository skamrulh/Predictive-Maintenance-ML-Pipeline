"""
Evaluation script for classification model performance.
Calculates comprehensive metrics including ROC-AUC, precision, recall, F1-score.
"""

import pandas as pd
import argparse
import yaml
import joblib
import json
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           precision_recall_curve, f1_score, precision_score, 
                           recall_score, confusion_matrix)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_optimal_threshold(y_true, probabilities):
    """Find optimal classification threshold that maximizes F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    #thresholds = np.append(thresholds, 1.0)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return float(thresholds[np.nanargmax(f1_scores)])


def main(config_path="config.yaml"):
    """Main evaluation function for classification model."""
    print("Evaluating classification model performance...")
    
    # Load configuration
    cfg = load_config(config_path)
    
    # Load validation data
    df = pd.read_csv(cfg["paths"]["tabular_valid"])
    y_true = df["label_fail_within_24h"].values
    
    # Prepare features
    drop_cols = ["label_fail_within_24h", "machine_id", "hour"]
    X = df.drop(columns=drop_cols)
    
    print(f"Validation data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive samples: {y_true.sum()} ({y_true.sum()/len(y_true)*100:.2f}%)")
    
    # Load trained model
    try:
        pipe = joblib.load(cfg["paths"]["model_cls"])
        print("Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return
    
    # Generate predictions
    probabilities = pipe.predict_proba(X)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_true, probabilities)
    ap = average_precision_score(y_true, probabilities)
    optimal_threshold = find_optimal_threshold(y_true, probabilities)
    y_pred = (probabilities >= optimal_threshold).astype(int)
    
    # Calculate comprehensive metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Prepare output
    output = {
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "threshold": optimal_threshold,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist()
    }
    
    # Save metrics
    with open(cfg["paths"]["metrics_cls"], "w") as f:
        json.dump(output, f, indent=2)
    
    # Print results
    print(f"\n📊 CLASSIFICATION MODEL RESULTS:")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC: {ap:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Confusion Matrix: {cm.tolist()}")
    print(f"Metrics saved: {cfg['paths']['metrics_cls']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification model")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    main(args.config)