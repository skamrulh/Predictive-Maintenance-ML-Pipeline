"""
Common utilities and model definitions for predictive maintenance system.
Contains preprocessing pipelines, model configurations, and shared functions.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import pandas as pd
import numpy as np
from typing import List, Union


def pre_cls(cols: List[str]) -> Pipeline:
    """
    Create classification pipeline.
    
    Args:
        cols (List[str]): List of column names to be scaled and used by the model.
    """
    return Pipeline([
        ("pre", ColumnTransformer([("num", StandardScaler(), cols)])), 
        ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=123, n_jobs=-1))
    ])


def pre_rul(cols: List[str]) -> Pipeline:
    """
    Create RUL prediction pipeline.
    
    Args:
        cols (List[str]): List of column names to be scaled and used by the model.
    """
    return Pipeline([
        ("pre", ColumnTransformer([("num", StandardScaler(), cols)])), 
        ("model", GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=123))
    ])


def build_rul(stream: pd.DataFrame, cap: int = 500) -> pd.DataFrame:
    """
    Build Remaining Useful Life (RUL) dataset from sensor stream data.
    
    This function calculates the RUL for every record by finding the time
    difference between the current hour and the next failure event, capping the RUL.
    
    Args:
        stream (pd.DataFrame): Raw sensor data with machine_id, hour, and failure columns
        cap (int): Maximum RUL value to cap predictions (e.g., 500 hours)
        
    Returns:
        pd.DataFrame: Dataset with the added 'RUL_hours' label column.
    """
    stream = stream.sort_values(["machine_id", "hour"]).reset_index(drop=True)
    out: List[Union[int, float]] = []
    
    for _, g in stream.groupby("machine_id"):
        fails = g.loc[g.failure == 1, "hour"].values
        k = 0
        for _, r in g.iterrows():
            h = r.hour
            # Skip past failures that occurred before the current hour (h)
            while k < len(fails) and fails[k] < h:
                k += 1
            
            # If no more failures exist, RUL is capped. Otherwise, calculate RUL.
            if k >= len(fails):
                rul = cap
            else:
                rul = max(0, fails[k] - h)
                
            out.append(min(cap, rul))
    
    result = stream.copy()
    result["RUL_hours"] = out
    return result