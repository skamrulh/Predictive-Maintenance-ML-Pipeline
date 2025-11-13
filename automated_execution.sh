# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete project automatically
python run_project.py

# 3. Start the API
uvicorn api:app --reload --host 0.0.0.0 --port 8000