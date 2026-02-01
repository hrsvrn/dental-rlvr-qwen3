import os
import io
import torch
from huggingface_hub import snapshot_download
from datasets import load_dataset, Image as HFImage
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==========================================
# PART 1: ROBUST DENTEX LOADER
# ==========================================
def get_dentex_dataset(local_dir="./DENTEX_LOCAL"):
    local_base = os.path.join(local_dir, "DENTEX")
    if not os.path.exists(local_base):
        print(f"Downloading dataset to {local_dir}...")
        snapshot_download(repo_id="ibrahimhamamci/DENTEX", repo_type="dataset", 
                          local_dir=local_dir, local_dir_use_symlinks=False)
    
    data_files = {"train": os.path.join(local_base, "training_data.zip")}

    print(f"--- Loading and Scanning Dataset ---")
    ds = load_dataset("imagefolder", data_files=data_files, split="train")
    ds = ds.cast_column("image", HFImage(decode=False))

    def is_real_image(example):
        path = example["image"].get("path", "").lower()
        # Suffix-aware check for the specific ZIP internal structure
        return any(ext in path for ext in ['.png', '.jpg', '.jpeg']) and ".json" not in path

    ds_valid = ds.filter(is_real_image)
    ds_valid = ds_valid.cast_column("image", HFImage(decode=True))
    print(f"Successfully loaded {len(ds_valid)} valid dental images.")
    return ds_valid

# ==========================================
# PART 2: QWEN BASELINE INFERENCE
# ==========================================
def run_baseline(dataset, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
    print(f"--- Initializing Model: {model_id} ---")
    
    # Load model with automatic precision and device mapping
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Pick the first image from your 3603 valid images
    sample = dataset[0]
    image = sample["image"]
    
    print(f"Testing inference on image: {getattr(image, 'filename', 'Memory Object')}")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the dental findings in this X-ray. Be specific about tooth enumeration and any visible caries."}
        ],
    }]

    # Pre-process for Qwen
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, 
        padding=True, return_tensors="pt"
    ).to(model.device)

    # Generate output
    print("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    trimmed_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    print("\n" + "="*30)
    print("QWEN ANALYSIS:")
    print(output_text)
    print("="*30)

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load the dataset (3603 images)
    ds = get_dentex_dataset()
    
    if ds and len(ds) > 0:
        # 2. Run the Qwen test
        run_baseline(ds)