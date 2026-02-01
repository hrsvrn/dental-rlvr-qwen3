import os
import io
from huggingface_hub import snapshot_download
from datasets import load_dataset, Image as HFImage
from PIL import Image

def get_dentex_dataset(local_dir="./DENTEX_LOCAL"):
    # 1. DOWNLOAD PHASE
    local_base = os.path.join(local_dir, "DENTEX")
    if not os.path.exists(local_base):
        print(f"Downloading dataset to {local_dir}...")
        snapshot_download(
            repo_id="ibrahimhamamci/DENTEX",
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    
    # 2. CONFIGURATION
    data_files = {
        "train": os.path.join(local_base, "training_data.zip"),
        "validation": os.path.join(local_base, "validation_data.zip")
    }

    print(f"--- Loading and Scanning Entire Dataset ---")
    try:
        # Load the split
        ds = load_dataset("imagefolder", data_files=data_files, split="train")
        
        # Cast to raw bytes to allow path string manipulation
        ds = ds.cast_column("image", HFImage(decode=False))

        def is_real_image(example):
            # Based on your debug: zip://path/to/image.png::/absolute/path/to.zip
            path = example["image"].get("path", "").lower()
            
            # Check for image extensions before the "::" separator
            has_img_ext = any(ext in path for ext in ['.png', '.jpg', '.jpeg'])
            # Ensure we don't accidentally pick up the .json files found in your debug
            is_not_json = ".json" not in path
            # Exclude system-generated junk
            is_not_junk = "__macosx" not in path
            
            return has_img_ext and is_not_json and is_not_junk

        print("Scanning for valid X-rays using suffix-aware filtering...")
        ds_valid = ds.filter(is_real_image)
        
        if len(ds_valid) == 0:
            print("Still no images found. Checking first path seen:")
            print(f"Path: {ds[0]['image']['path']}")
            return None

        # Re-enable PIL decoding
        ds_valid = ds_valid.cast_column("image", HFImage(decode=True))
        
        print(f"Successfully loaded {len(ds_valid)} valid dental images.")
        return ds_valid

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    dataset = get_dentex_dataset()
    if dataset:
        print(f"Success! Found {len(dataset)} images.")
        print(f"Sample Image Size: {dataset[0]['image'].size}")