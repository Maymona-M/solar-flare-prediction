import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime

def main():
    print("=" * 50)
    print("Solar Flare Prediction Training Job")
    print(f"Start time: {datetime.now()}")
    print("=" * 50)
    
    # Check if data path is provided
    data_path = os.getenv("AZUREML_DATASET_INPUT_training_data", "/tmp/data")
    print(f"Data path: {data_path}")
    
    # List files in data path
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        print(f"Files in data path: {files}")
        
        # Look for CSV file
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(os.path.join(data_path, csv_files[0]))
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First few rows:\n{df.head()}")
    else:
        print(f"Data path {data_path} does not exist")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create a dummy model output
    model_output_path = os.getenv("AZUREML_OUTPUT_model_output", "./outputs")
    os.makedirs(model_output_path, exist_ok=True)
    
    # Save a dummy model file
    dummy_model = {"status": "training_complete", "timestamp": str(datetime.now())}
    import json
    with open(os.path.join(model_output_path, "model_info.json"), "w") as f:
        json.dump(dummy_model, f)
    
    print(f"Model output saved to {model_output_path}")
    print("Training job completed successfully!")

if __name__ == "__main__":
    main()
