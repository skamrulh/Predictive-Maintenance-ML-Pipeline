# Predictive Maintenance ML System

This project is an end-to-end machine learning system for predictive maintenance. It ingests synthetic sensor data, trains two separate models (one for failure classification and one for RUL regression), and serves these models via a live FastAPI REST API.

## Key Features

* **Failure Classification:** A `RandomForestClassifier` predicts the probability of an equipment failure within the next 24 hours.
* **RUL Regression:** A `GradientBoostingRegressor` predicts the Remaining Useful Life (RUL) of the equipment in hours.
* **Synthetic Data:** A data generation script (`generate_synthetic_data.py`) creates a realistic sensor stream with failure events.
* **End-to-End Pipeline:** Full pipeline from data generation and training to evaluation.
* **API Server:** A `FastAPI` application serves the trained models for real-time predictions.

## 📁 Project Structure

Project/
├── data/ # (Generated) Raw sensor stream & tabular data
├── models/ # (Generated) Trained .joblib model files
├── artifacts/ # (Generated) Model evaluation metrics (.json)
│
├── api.py # FastAPI application
├── common.py # Shared functions (model pipelines, RUL logic)
├── config.yaml # Configuration file for paths and parameters
│
├── generate_synthetic_data.py # 1. Run this first
├── train_classification.py # 2. Run this second (Not provided)
├── train_rul.py # 3. Run this third (Not provided)
│
├── evaluate_classification.py # (Optional) Evaluate the classification model
├── evaluate_rul.py # (Optional) Evaluate the RUL model
├── predict_classification.py # Example batch prediction script
│
└── requirements.txt # Python dependencies