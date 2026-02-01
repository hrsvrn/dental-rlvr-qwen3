import os
import io
from huggingface_hub import snapshot_download
from datasets import load_dataset, Image as HFImage
from PIL import Image

def get_dentex_dataset(split="train", local_dir="./DENTEX_LOCAL"):
    """
    Downloads the dataset to a local folder and loads it into a 
    Hugging Face Dataset object with robust image filtering.
    """
    
    # 1. DOWNLOAD PHASE
    # Check if the training zip exists to avoid re-downloading
    expected_zip = os.path.join(local_dir, "DENTEX/training_data.zip")
    
    if not os.path.exists(expected_zip):
        print(f"Dataset not found at {local_dir}. Starting download...")
        snapshot_download(
            repo_id="ibrahimhamamci/DENTEX",
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Physically copies files to the folder
        )
        print("Download complete.")
    else:
        print(f"Local dataset found at {local_dir}. Skipping download.")

    # 2. LOADING PHASE
    local_base = os.path.join(local_dir, "DENTEX")
    data_files = {
        "train": os.path.join(local_base, "training_data.zip"),
        "validation": os.path.join(local_base, "validation_data.zip"),
        "test": os.path.join(local_base, "test_data.zip")
    }

    print(f"--- Loading Dentex Split: {split} ---")
    try:
        # Load using the imagefolder builder
        ds = load_dataset("imagefolder", data_files=data_files, split=split)
        
        # Disable automatic decoding to safely filter non-image entries
        ds = ds.cast_column("image", HFImage(decode=False))

        def is_real_image(example):
            path = example["image"].get("path", "").lower()
            # Verify file extension and exclude system-generated junk
            is_img = path.endswith(('.png', '.jpg', '.jpeg'))
            is_not_metadata = "__macosx" not in path and ".ds_store" not in path
            return is_img and is_not_metadata

        print("Filtering for valid dental X-ray files...")
        ds = ds.filter(is_real_image)
        
        # Enable PIL decoding for the model to use
        ds = ds.cast_column("image", HFImage(decode=True))
        
        print(f"Successfully loaded {len(ds)} valid images.")
        return ds

    except Exception as e:
        print(f"Error during loading: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    # First run will download 11GB+; subsequent runs will be near-instant.
    dataset = get_dentex_dataset(split="train[:100]")
    
    if dataset:
        print(f"Dataset columns: {dataset.column_names}")
        print(f"First image metadata: {dataset[0]['image']}")