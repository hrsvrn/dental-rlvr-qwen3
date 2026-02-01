import torch
from unsloth import FastVisionModel
from transformers import AutoProcessor
from PIL import Image
import requests
from io import BytesIO

# Configuration
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

def load_inference_engine():
    print("Loading model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        device_map="auto",
    )
    FastVisionModel.for_inference(model) # Enable native 2x faster inference
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, tokenizer, processor

def run_inference(model, processor, image_input, prompt_text="Identify the dental condition visible in this X-ray."):
    """
    Runs inference on a single image.
    image_input: PIL Image object or path string.
    """
    # Load image if string
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            response = requests.get(image_input)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_input)
    else:
        image = image_input

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            use_cache=True,
            temperature=0.2
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

from dentex_loader import get_dentex_dataset
from datasets import Dataset

if __name__ == "__main__":
    # Load Model
    model, _, processor = load_inference_engine()
    print("Engine loaded.")
    
    # Load Dataset
    ds = get_dentex_dataset(split="train")
    if not ds:
        print("Could not load Dentex dataset.")
        exit()
        
    print(f"Loaded {len(ds)} samples.")
    
    # Run Inference on first 3 samples
    for i in range(3):
        example = ds[i]
        image = example["image"]
        # Ground truth might be in 'label', 'text', or 'diagnosis' column. 
        # Inspecting keys based on previous knowledge or just dumping all
        ground_truth = str(example) 
        
        print(f"\n--- Sample {i} ---")
        # print(f"Ground Truth Data: {ground_truth[:200]}...") # truncate for display
        
        result = run_inference(model, processor, image)
        print(f"Prediction: {result}")

