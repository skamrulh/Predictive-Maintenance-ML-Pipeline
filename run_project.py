"""
Automated script to run the entire predictive maintenance project.
Executes all steps in the correct order with error handling.
"""

import os
import sys
import subprocess
import time


def run_command(description, command):
    """Run a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error:")
        print(f"Error: {e.stderr}")
        print(f"Return code: {e.returncode}")
        return False


def main():
    """Main execution function."""
    print("🤖 Starting Automated Predictive Maintenance Project Execution")
    print("This will run all steps in sequence...")
    
    steps = [
        ("Generating synthetic data", "python generate_synthetic_fixed.py"),
        ("Training classification model", "python train_classification_fixed.py --config config_fixed.yaml"),
        ("Training RUL model", "python train_rul_fixed.py --config config_fixed.yaml"),
        ("Evaluating classification model", "python evaluate_classification_fixed.py --config config_fixed.yaml"),
        ("Evaluating RUL model", "python evaluate_rul_fixed.py --config config_fixed.yaml"),
    ]
    
    # Execute all steps
    for description, command in steps:
        success = run_command(description, command)
        if not success:
            print(f"\n💥 Execution stopped due to failure in: {description}")
            sys.exit(1)
        time.sleep(1)  # Brief pause between steps
    
    # Final summary
    print(f"\n{'🎉' * 20}")
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"{'🎉' * 20}")
    
    print("\n📁 Generated Files:")
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(ext in file for ext in ['.csv', '.joblib', '.json', '.yaml']):
                print(f"  📄 {os.path.join(root, file)}")
    
    print("\n🎯 Next Steps:")
    print("1. Start the API server:")
    print("   uvicorn api_fixed:app --reload --host 0.0.0.0 --port 8000")
    print("2. Access API documentation at: http://localhost:8000/docs")
    print("3. Test the API with the provided examples")


if __name__ == "__main__":
    main()