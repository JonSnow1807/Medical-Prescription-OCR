#!/usr/bin/env python3
"""
Download the Medical Prescription OCR model from Hugging Face.
"""

import os
import sys
from huggingface_hub import snapshot_download

def download_model(model_dir="model"):
    """Download model files from Hugging Face Hub."""
    
    repo_id = "chinmays18/medical-prescription-ocr"
    
    print(f"Downloading model from {repo_id}...")
    print(f"This will download ~800MB of model files to '{model_dir}/'")
    
    try:
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download all model files
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"\n✅ Model downloaded successfully to {model_dir}/")
        print("You can now run: python app.py")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nPlease check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
