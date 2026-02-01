import torch
from datasets import load_dataset
from unsloth import FastVisionModel, is_bfloat16_supported
from trl import GRPOTrainer, GRPOConfig
from dentex_loader import get_dentex_dataset
from transformers import AutoProcessor

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_DIR = "qwen2-vl-dentex-unsloth-rlvr"
MAX_SEQ_LENGTH = 1024 # Unsloth handles RoPE scaling automatically
LORA_RANK = 16
LORA_ALPHA = 32

def reward_function(prompts, completions, **kwargs):
    """
    Reward function for GRPO.
    Checks if the completion contains the correct diagnosis.
    Expects 'label' in kwargs (passed from dataset map).
    """
    rewards = []
    ground_truths = kwargs.get("label") 
    
    for completion, gt in zip(completions, ground_truths):
        c_norm = completion.lower()
        gt_norm = str(gt).lower()
        
        # Simple inclusion check 
        if gt_norm in c_norm:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

def main():
    print("Loading Unsloth Qwen2-VL model...")
    
    # 1. Load Model with Unsloth
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True, # QLoRA
        use_gradient_checkpointing="unsloth", # optimized GC
        device_map="auto",
    )
    
    # 2. Add LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True, # Enable vision layer finetuning
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Load & Process Dataset
    print("Loading dataset...")
    raw_dataset = get_dentex_dataset(split="train")
    if not raw_dataset:
        print("Dataset failed to load. Using mock data just to verify loop if needed, or exiting.")
        return

    # Processor for formatting prompt (Qwen2-VL needs specific chat template)
    # Unsloth 'tokenizer' might be the processor or just tokenizer.
    # For VLMs, we usually need the processor.
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    def process_data(example):
        # Construct message for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": "Identify the dental condition visible in this X-ray."}
                ]
            }
        ]
        # Get the text prompt (without generation)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "image": example["image"], 
            "label": str(example.get("label", "Unknown"))
        }

    dataset = raw_dataset.map(process_data)

    # 4. GRPO Configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=MAX_SEQ_LENGTH,
        max_completion_length=128,
        num_generations=4, # Number of samples for GRPO
        beta=0.1,
        logging_step=1,
        save_steps=100,
        optim="adamw_8bit", # Optimize optimizer memory
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        report_to="none",
        remove_unused_columns=False, # Essential for passing 'image', 'label' etc that aren't model inputs directly? 
        # Actually GRPOTrainer needs to handle 'image'. 
        # If 'processing_class' handles it, TRL might work. Use 'processor'.
    )

    # 5. Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
        # peft_config=... model is already PEFT
    )

    print("Starting Training...")
    trainer.train()
    
    # Save
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="lora")

if __name__ == "__main__":
    main()
