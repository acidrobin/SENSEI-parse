"""
models.py

Functions for loading pretrained models, tokenizers, and applying LoRA PEFT.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from config import get_bnb_config, get_peft_config

def load_model_and_tokenizer(model_name="Llama-2-7b-hf", lora_dropout=0.5):
    """
    Load pretrained model with 4-bit quantization and LoRA PEFT, along with tokenizer.

    Args:
        model_name (str): Name of the pretrained model.
        lora_dropout (float): Dropout rate for LoRA layers.

    Returns:
        tuple: (model, tokenizer, peft_config)
    """
    # Quantization + PEFT configs
    bnb_config = get_bnb_config()
    peft_config = get_peft_config(lora_dropout)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        f"models/{model_name}",
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = get_peft_model(model, peft_config)

    # Model configuration tweaks
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config
