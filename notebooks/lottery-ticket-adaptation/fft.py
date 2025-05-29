import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# 1. Load UltraFeedback
ds = load_dataset("openbmb/UltraFeedback", split="train")  # 63,967 examples

# 2. Pick best completion by instruction_following Rating
def pick_best(example):
    best = max(
        example["completions"],
        key=lambda c: float(c["annotations"]["instruction_following"]["Rating"])
    )
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
    toks = tokenizer(text, truncation=False)  # no truncation
    toks["labels"] = toks["input_ids"].copy()
    return toks

tok_ds = proc.map(preprocess, remove_columns=["prompt", "response"])

# 5. Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="mistral-ultrafeedback",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-7,
    logging_steps=20,
    save_strategy="no",
    optim="paged_adamw_32bit",
    report_to="none",
)

# 7. Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
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