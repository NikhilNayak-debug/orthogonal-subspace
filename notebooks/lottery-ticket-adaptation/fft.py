import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. Load UltraFeedback
ds = load_dataset("openbmb/UltraFeedback", split="train")  # 63,967 examples

ds = ds.filter(lambda ex: ex.get("completions") is not None and len(ex["completions"]) > 0)

# 2. Pick best completion by instruction_following Rating
def pick_best(example):
    def get_if_score(c):
        r = c["annotations"]["instruction_following"]["Rating"]
        try:
            return float(r)
        except (ValueError, TypeError):
            return 0.0   # treat "N/A" (or missing) as 0.0

    best = max(example["completions"], key=get_if_score)
    return {
        "prompt": example["instruction"],
        "response": best["response"]
    }

proc = ds.map(pick_best, remove_columns=ds.column_names)

# 3. Tokenizer & model setup
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # ensure padding works

# 4. Preprocess WITHOUT truncation, adding EOS at end of each part
def preprocess(ex):
    eos = tokenizer.eos_token
    text = (
        f"{ex['prompt']}{eos}"
        f"{ex['response']}{eos}"
    )
    toks = tokenizer(text, truncation=True, max_length=1024, padding="max_length", add_special_tokens=False)  # no truncation
    input_ids = toks["input_ids"]
    # mask pad tokens in the labels
    labels = [
        (token if token != tokenizer.pad_token_id else -100)
        for token in input_ids
    ]
    toks["labels"] = labels
    return toks

tok_ds = proc.map(preprocess, remove_columns=["prompt", "response"])

# 7. Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
)

# 5. Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,               # causal LM
)

# model.gradient_checkpointing_enable()

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="mistral-ultrafeedback",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=5e-7,
    logging_steps=50,
    save_strategy="no",
    optim="adamw_torch",
    report_to="none",
    bf16=True,                        # keep bf16
    fsdp="full_shard auto_wrap",
    fsdp_config={
      "fsdp_transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
    },
)

# 8. Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("mistral-ultrafeedback-final")