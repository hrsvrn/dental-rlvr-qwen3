import os
import io
from datasets import load_dataset, Image as HFImage
from PIL import Image

def get_dentex_dataset(split="train"):
    # Path to the local files we just downloaded
    local_base = "./dentex_data/DENTEX"
    
    data_files = {
        "train": f"{local_base}/training_data.zip",
        "validation": f"{local_base}/validation_data.zip",
        "test": f"{local_base}/test_data.zip"
    }

    # Verify files exist before loading
    for name, path in data_files.items():
        if not os.path.exists(path):
            print(f"Error: Could not find {path}. Did you run the download step?")
            return None

    print(f"--- Loading Dentex from Local Folder ---")
    try:
        # We use the 'imagefolder' builder but point it to our local ZIPs
        ds = load_dataset("imagefolder", data_files=data_files, split=split)
        
        # Cast to raw bytes to filter out folder headers/junk
        ds = ds.cast_column("image", HFImage(decode=False))

        def is_valid_image(example):
            img_dict = example["image"]
            if not img_dict.get("bytes"):
                return False
            try:
                # verify() is fast; it only checks the file header
                Image.open(io.BytesIO(img_dict["bytes"])).verify()
                return True
            except Exception:
                return False

        print(f"Cleaning local dataset (removing non-image entries)...")
        ds = ds.filter(is_valid_image)
        
        # Re-enable PIL decoding for the model
        ds = ds.cast_column("image", HFImage(decode=True))
        
        print(f"Successfully loaded {len(ds)} valid images.")
        return ds

    except Exception as e:
        print(f"Failed to load local dataset: {e}")
        return None

if __name__ == "__main__":
    # Testing with the first 50 images
    ds = get_dentex_dataset(split="train[:50]")
    if ds:
        print(f"Sample Image Size: {ds[0]['image'].size}")