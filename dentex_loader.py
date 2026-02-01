import io
from datasets import load_dataset, Image as HFImage
from PIL import Image

def get_dentex_dataset(split="train"):
    print(f"--- Initializing Robust Dentex Loader ---")
    
    data_files = {
        "train": "DENTEX/training_data.zip",
        "validation": "DENTEX/validation_data.zip",
        "test": "DENTEX/test_data.zip"
    }

    try:
        # 1. Load the dataset 
        ds = load_dataset(
            "ibrahimhamamci/DENTEX", 
            data_files=data_files, 
            split=split
        )
        
        # 2. Turn OFF automatic decoding for the 'image' column to prevent crashing
        # This allows us to inspect the bytes before PIL tries to identify them.
        ds = ds.cast_column("image", HFImage(decode=False))
        
        print(f"Initial count: {len(ds)} items. Cleaning dataset...")

        def is_valid_image(example):
            try:
                img_bytes = example["image"]["bytes"]
                # Try to open and verify the image headers
                Image.open(io.BytesIO(img_bytes)).verify()
                return True
            except Exception:
                return False

        # 3. Filter out non-image files (.DS_Store, metadata, etc.)
        ds = ds.filter(is_valid_image)
        
        # 4. Turn decoding back ON for the valid images
        ds = ds.cast_column("image", HFImage(decode=True))
        
        print(f"Final valid image count: {len(ds)}")
        return ds
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        return None

if __name__ == "__main__":
    # Test with a small slice
    ds = get_dentex_dataset(split="train[:20]")
    if ds and len(ds) > 0:
        print(f"Success! Sample image format: {ds[0]['image'].format}")
        print(f"Sample image size: {ds[0]['image'].size}")