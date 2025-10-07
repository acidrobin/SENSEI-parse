"""
download_model.py

Download a pretrained model and tokenizer, apply LoRA configuration,
and save them locally. No training is performed.
"""

import os
import os.path as op
from transformers import AutoModelForPreTraining, AutoTokenizer
from peft import get_peft_model
from config import get_peft_config

# ----------------------
# Configuration
# ----------------------
MODEL_NAME = "flan-t5-xl"
SAVE_DIR = f"models/{MODEL_NAME}"
LORA_DROPOUT = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------
# Load model and tokenizer
# ----------------------
model = AutoModelForPreTraining.from_pretrained(f"google/{MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(f"google/{MODEL_NAME}", trust_remote_code=True)

# Optional: adjust tokenizer padding token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ----------------------
# Apply LoRA
# ----------------------
peft_config = get_peft_config(LORA_DROPOUT)
model = get_peft_model(model, peft_config)

# ----------------------
# Save locally
# ----------------------
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model with LoRA config and tokenizer saved to {SAVE_DIR}")
