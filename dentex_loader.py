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
            local_dir_use_symlinks=False # Physical copy
        )
    
    # 2. CONFIGURATION
    data_files = {
        "train": os.path.join(local_base, "training_data.zip"),
        "validation": os.path.join(local_base, "validation_data.zip")
    }

    print(f"--- Loading and Scanning Entire Dataset ---")
    try:
        # Load the FULL train split first to find where the images actually start
        ds = load_dataset("imagefolder", data_files=data_files, split="train")
        
        # Cast to raw bytes for path inspection
        ds = ds.cast_column("image", HFImage(decode=False))

        def is_real_image(example):
            path = example["image"].get("path", "").lower()
            # We want actual image files, skipping the folder headers
            return path.endswith(('.png', '.jpg', '.jpeg')) and "__macosx" not in path

        # Filter the whole dataset to find ONLY the images
        print("Scanning ZIP contents for valid X-rays (this may take a minute)...")
        ds_valid = ds.filter(is_real_image)
        
        if len(ds_valid) == 0:
            print("DEBUG: No images found. First 5 paths seen in ZIP:")
            for i in range(min(5, len(ds))):
                print(f"  - {ds[i]['image']['path']}")
            return None

        # Re-enable decoding for the valid subset
        ds_valid = ds_valid.cast_column("image", HFImage(decode=True))
        
        print(f"Successfully loaded {len(ds_valid)} valid dental images.")
        return ds_valid

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    dataset = get_dentex_dataset()
    if dataset:
        # Now you can safely slice the VALID images
        sample_slice = dataset.select(range(10))
        print(f"First valid image size: {sample_slice[0]['image'].size}")