import os
from datasets import load_dataset
from PIL import Image
import io

def is_valid_image(example):
    """Checks if the image can actually be opened by PIL."""
    try:
        # The 'image' column contains a dict with 'bytes' or a PIL object
        img_data = example.get("image")
        if img_data is None:
            return False
            
        # If it's already a PIL object, it's valid
        if isinstance(img_data, Image.Image):
            return True
            
        # If it's bytes, try to open it
        if isinstance(img_data, dict) and "bytes" in img_data:
            Image.open(io.BytesIO(img_data["bytes"])).verify()
        
        return True
    except Exception:
        return False

def get_dentex_dataset(split="train"):
    print(f"--- Initializing Robust Dentex Loader ---")
    
    data_files = {
        "train": "DENTEX/training_data.zip",
        "validation": "DENTEX/validation_data.zip",
        "test": "DENTEX/test_data.zip"
    }

    try:
        # Load without immediate decoding to prevent the identification error
        ds = load_dataset(
            "ibrahimhamamci/DENTEX", 
            data_files=data_files, 
            split=split,
            decode=False # <--- Key change: don't decode images immediately
        )
        
        print(f"Initial count: {len(ds)} items. Cleaning dataset...")
        
        # Filter out non-image files (like .txt, .json, or hidden system files)
        ds = ds.filter(is_valid_image)
        
        # Now that it's clean, we re-enable decoding
        ds = ds.with_format("pil")
        
        print(f"Final valid image count: {len(ds)}")
        return ds
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        return None

if __name__ == "__main__":
    ds = get_dentex_dataset(split="train[:50]")
    if ds and len(ds) > 0:
        print("Success! First image size:", ds[0]["image"].size)