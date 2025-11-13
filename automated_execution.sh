# -----------------------------------------------------------
# Full Project Execution Script
# Runs data generation, training, evaluation, and starts the API
# -----------------------------------------------------------

# 1. Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 2. Run the complete pipeline sequentially
echo "Starting full Predictive Maintenance pipeline..."

# 2a. Data Generation
python generate_synthetic_data.py

# 2b. Training
python train_classification.py
python train_rul.py

# 2c. Evaluation
python evaluate_classification.py
python evaluate_rul.py

echo "✅ Pipeline steps completed successfully."

# 3. Start the API
echo "Starting FastAPI server..."
# Using the correct file name 'api:app' and port
uvicorn api:app --reload --host 0.0.0.0 --port 8000
