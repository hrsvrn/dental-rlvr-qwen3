import os
from datasets import load_dataset
from PIL import Image

def get_dentex_dataset(split="train"):
    """
    Loads the Dentex dataset by pointing directly to the ZIP archives 
    found in the ibrahimhamamci/DENTEX repository.
    """
    print(f"--- Initializing Dentex Loader ---")
    print(f"Target split: {split}")
    
    # Define the mapping based on the file structure in your screenshots
    # The 'imagefolder' loader will automatically extract these zips to your cache.
    data_files = {
        "train": "DENTEX/training_data.zip",
        "validation": "DENTEX/validation_data.zip",
        "test": "DENTEX/test_data.zip"
    }

    try:
        # We specify 'imagefolder' to ensure it treats the ZIP contents as images
        ds = load_dataset(
            "ibrahimhamamci/DENTEX", 
            data_files=data_files, 
            split=split
        )
        
        print(f"Successfully loaded {len(ds)} samples.")
        print(f"Available columns: {ds.column_names}")
        
        # Basic sanity check: ensure 'image' column exists and is not empty
        if "image" in ds.column_names:
            ds = ds.filter(lambda x: x["image"] is not None)
        
        return ds
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset.")
        print(f"Details: {e}")
        print("\nTip: Ensure you have enough disk space in ~/.cache/huggingface")
        print("as the training zip is ~11GB and needs to be extracted.")
        return None

if __name__ == "__main__":
    # Test with a small slice to avoid massive extraction if just testing logic
    # Note: The first run will still download the full ZIP.
    dataset_sample = get_dentex_dataset(split="train[:5]")
    
    if dataset_sample:
        sample = dataset_sample[0]
        print("\n--- Sample Metadata ---")
        print(f"Image Type: {type(sample['image'])}")
        
        # If the folder structure inside the zip has labels (e.g. /train/caries/img1.png)
        # 'imagefolder' will automatically create a 'label' column.
        if "label" in sample:
            label_names = dataset_sample.features["label"].names
            print(f"Label: {label_names[sample['label']]} ({sample['label']})")
        
        # sample['image'].show() # Uncomment to pop open the X-ray image