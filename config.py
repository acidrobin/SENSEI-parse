"""
config.py

Contains reusable configuration functions for LoRA and BitsAndBytes
quantization setups for model training.
"""

from peft import LoraConfig
from transformers import BitsAndBytesConfig

def get_bnb_config():
    """
    Return BitsAndBytes 4-bit quantization configuration.

    Returns:
        BitsAndBytesConfig: Configuration object for 4-bit quantization.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=False,
    )

def get_peft_config(lora_dropout):
    """
    Return LoRA PEFT configuration for causal language models.

    Args:
        lora_dropout (float): Dropout rate for LoRA layers.

    Returns:
        LoraConfig: LoRA configuration object.
    """
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=lora_dropout,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
    )
