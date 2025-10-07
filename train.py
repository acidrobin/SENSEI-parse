"""
train.py

Main training script that orchestrates loading datasets, model, tokenizer,
callbacks, and starts training using SFTTrainer.
"""

import os
import os.path as op
from argparse import ArgumentParser

from trl import SFTTrainer
from transformers import TrainingArguments
from callbacks import EvalCallback
from datasets import load_train_dataset, load_val_dataset
from models import load_model_and_tokenizer

# ----------------------
# Argument parsing
# ----------------------
parser = ArgumentParser()
parser.add_argument("--mode", type=str, default="debatabase")
parser.add_argument("--test_set", type=str, default="debatabase")
parser.add_argument("--lora_dropout", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

# ----------------------
# Directories
# ----------------------
subdir_name = f"lr_{args.learning_rate}_wd_{args.weight_decay}_dropout_{args.lora_dropout}"
if args.debug:
    subdir_name += "_DEBUG"

scores_dir = op.join("scores", args.test_set, args.mode, subdir_name)
finetuned_dir = op.join("models_finetuned", args.test_set, args.mode, subdir_name)
os.makedirs(scores_dir, exist_ok=True)

# ----------------------
# Datasets
# ----------------------
train_dataset = load_train_dataset(args.mode.split("_"), debug=args.debug)
val_dataset = load_val_dataset(args.test_set, test=False, debug=args.debug)

# ----------------------
# Model + Tokenizer
# ----------------------
model_name = "Llama-2-7b-hf"
model, tokenizer, peft_config = load_model_and_tokenizer(model_name, args.lora_dropout)

# ----------------------
# Training
# ----------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    optim="adamw_torch_fused",
    save_strategy="no",
    logging_steps=10,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

eval_callback = EvalCallback(
    val_dataset=val_dataset,
    tokenizer=tokenizer,
    model=model,
    scores_dir=scores_dir,
    model_name=model_name,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=7000,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
    callbacks=[eval_callback],
)

trainer.train()
trainer.model.save_pretrained(finetuned_dir)
