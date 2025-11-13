"""
Synthetic data generation for predictive maintenance system.
Creates realistic sensor data with failure patterns for model development.
"""

import numpy as np
import pandas as pd
import os


def generate_synthetic_data():
    """Generate complete synthetic dataset for predictive maintenance."""
    print("Generating synthetic predictive maintenance data...")
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(123)
    
    # Parameters
    M, H = 50, 1500  # 50 machines, 1500 hours each
    sensors = ["temp", "vibration", "pressure", "current", "sound"]
    rows = []
    
    print(f"Creating sensor stream for {M} machines over {H} hours...")
    
    for m in range(M):
        # Machine-specific baseline
        base = rng.normal(0, 0.5, len(sensors))
        
        # Generate failure times (1-2 failures per machine)
        fail = sorted(rng.choice(np.arange(300, H), size=rng.integers(1, 3), replace=False))
        flags = np.zeros(H, dtype=int)
        flags[fail] = 1
        
        for t in range(H):
            # Generate sensor values with trends and failure effects
            vals = []
            for i in range(len(sensors)):
                # Base value with noise
                base_val = base[i] + rng.normal(0, 1.0)
                
                # Gradual degradation over time
                degradation = 0.0008 * t
                
                # Failure effects (sensors respond before actual failure)
                failure_effect = sum(1/(1+np.exp(-(t-ft)/20)) for ft in fail) * (0.5 + 0.3*i)
                
                sensor_value = base_val + degradation + failure_effect
                vals.append(sensor_value)
            
            row = {
                "machine_id": m,
                "hour": t,
                **{s: v for s, v in zip(sensors, vals)},
                "failure": flags[t]
            }
            rows.append(row)
    
    # Create sensor stream dataframe
    stream = pd.DataFrame(rows)
    stream.to_csv("data/sensor_stream.csv", index=False)
    print(f"✓ Created sensor stream with {len(stream)} records")
    
    # Create feature windows for classification
    print("Creating feature windows for classification...")
    features = create_window_features(stream, window=12)
    
    # Split into train/validation/holdout
    train_df = features.sample(frac=0.7, random_state=123)
    temp_df = features.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=124)
    holdout_df = temp_df.drop(val_df.index)
    
    # Save all datasets
    train_df.to_csv("data/train_tabular.csv", index=False)
    val_df.to_csv("data/valid_tabular.csv", index=False)
    holdout_df.to_csv("data/holdout_tabular.csv", index=False)
    
    print("✓ Created tabular datasets:")
    print(f"  - Training: {len(train_df)} samples")
    print(f"  - Validation: {len(val_df)} samples") 
    print(f"  - Holdout: {len(holdout_df)} samples")
    print(f"  - Failure rate: {features['label_fail_within_24h'].mean()*100:.2f}%")
    
    return stream, features


def create_window_features(df, window=12):
    """
    Create time-window features from sensor stream data.
    
    Args:
        df (pd.DataFrame): Sensor stream data
        window (int): Rolling window size
        
    Returns:
        pd.DataFrame: Feature dataset with rolling statistics
    """
    sensors = ["temp", "vibration", "pressure", "current", "sound"]
    feats = []
    
    for machine_id, group in df.groupby("machine_id"):
        group = group.sort_values("hour").reset_index(drop=True)
        
        for t in range(window, len(group) - 1):
            # Get window of previous readings
            window_data = group.iloc[t - window:t]
            
            # Create target: failure in next 24 hours
            next_24h = group.iloc[t:t + 24]
            failure_in_24h = int(
                group.loc[t, "failure"] == 1 or 
                next_24h["failure"].any()
            )
            
            # Create feature row
            row = {
                "machine_id": machine_id,
                "hour": int(group.loc[t, "hour"]),
                "label_fail_within_24h": failure_in_24h
            }
            
            # Calculate rolling statistics for each sensor
            for col in sensors:
                arr = window_data[col].values
                row.update({
                    f"{col}_mean": float(np.mean(arr)),
                    f"{col}_std": float(np.std(arr)),
                    f"{col}_min": float(np.min(arr)),
                    f"{col}_max": float(np.max(arr)),
                    f"{col}_median": float(np.median(arr))
                })
            
            feats.append(row)
    
    return pd.DataFrame(feats)


if __name__ == "__main__":
    generate_synthetic_data()
    print("\n" + "="*50)
    print("SYNTHETIC DATA GENERATION COMPLETE!")
    print("="*50)