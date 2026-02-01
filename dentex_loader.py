from datasets import load_dataset
from PIL import Image
import io

def get_dentex_dataset(split="train"):
    print(f"Loading Dentex dataset split: {split}...")
    try:
        # Load from HF - using 'main' or specific config if needed
        # Structure of ibrahimhamamci/DENTEX might need 'viewer' or 'enumeration' config?
        # Defaulting to no config name which usually loads default
        ds = load_dataset("ibrahimhamamci/DENTEX", split=split)
        
        # Print columns to help debug if keys are wrong
        print(f"Dataset columns: {ds.column_names}")
        
        # Filter for valid images
        ds = ds.filter(lambda x: x["image"] is not None)
        
        return ds
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return a dummy dataset if real one fails (for testing flow without internet/auth)
        return None

if __name__ == "__main__":
    # Test loading
    ds = get_dentex_dataset(split="train[:10]")
    if ds:
        print("Successfully loaded dataset sample.")
