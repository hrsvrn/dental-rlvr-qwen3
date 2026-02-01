# Dental VLM: Qwen2-VL on Dentex Dataset

This project implements an inference engine and an RLVR (Reinforcement Learning with Verifiable Rewards) training pipeline for Dental X-ray analysis, leveraging the **Qwen2-VL-7B-Instruct** model and the **Dentex** dataset.

It uses the **Unsloth** framework for efficient 4-bit quantization and faster inference/training.

## Project Structure

- `inference.py`: **Inference Engine**. Loads Qwen2-VL (4-bit) and runs inference on samples from the Dentex dataset.
- `train_rlvr.py`: **Training Script**. Implements RLVR using TRL's `GRPOTrainer` and Unsloth's LoRA adapters.
- `dentex_loader.py`: **Dataset Loader**. Utility to load and filter the `ibrahimhamamci/DENTEX` dataset from Hugging Face.
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.10+
- GPU with CUDA support (Recommended: 24GB VRAM for training, less for inference with 4-bit)
- [Unsloth](https://github.com/unslothai/unsloth) installed (provides `FastVisionModel`).

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure Unsloth is installed correctly for your CUDA version.*

## Usage

### Inference

Run the inference engine to test the model on Dentex samples:

```bash
python inference.py
```

This will:
1. Load Qwen2-VL-7B-Instruct in 4-bit mode.
2. Download the Dentex dataset (train split).
3. Run inference on the first 3 images, asking the model to "Identify the dental condition".

### Training (RLVR)

To train the model using Group Relative Policy Optimization (GRPO):

```bash
python train_rlvr.py
```

*Note: You may need to adjust `LORA_RANK`, `BATCH_SIZE`, or `GRADIENT_ACCUMULATION` in the script based on your GPU memory.*
