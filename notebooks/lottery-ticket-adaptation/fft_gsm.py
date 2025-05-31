import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
)

tokenizer.pad_token = tokenizer.eos_token  # ensure padding works
model.config.pad_token_id = tokenizer.eos_token_id

# 1. Load GSM8K from local JSON file
ds = load_dataset("json", data_files={"train": "/workspace/lottery-ticket-adaptation/rlaif/tasks/gsm8k/train.json"}, split="train")

# 2. map each example through the official chat template
# def preprocess(example):
#     messages = [
#         {"role": "user",      "content": example["question"]},
#         {"role": "assistant", "content": example["answer"]},
#     ]
#     enc = tokenizer.apply_chat_template(
#         messages,
#         padding="longest",
#         truncation=True,
#         max_length=2048,
#         return_dict=True,        # <<< ADD THIS
#         add_generation_prompt=False,
#     )

#     return {
#         "input_ids":      enc["input_ids"],
#         "attention_mask": enc["attention_mask"],
#     }

def preprocess(example):
    messages = [
        {"role": "user",      "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]

    # apply_chat_template as before
    enc = tokenizer.apply_chat_template(
        messages,
        padding="longest",
        truncation=True,
        max_length=2048,
        return_dict=True,
        add_generation_prompt=False,
    )

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # 1a) find the token‐ID for the closing marker "[/INST]"
    sep_id = tokenizer.convert_tokens_to_ids("[/INST]")

    # 1b) locate the last occurrence of sep_id in input_ids
    #     (that tells us where the question block ends)
    inst_end_idx = 0
    for i, tok in enumerate(input_ids):
        if tok == sep_id:
            inst_end_idx = i

    # 1c) build labels: mask everything <= inst_end_idx to -100, leave answer tokens untouched
    labels = []
    for idx, tok in enumerate(input_ids):
        if idx <= inst_end_idx:
            labels.append(-100)
        else:
            labels.append(tok)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

tokenized = ds.map(preprocess, batched=False)

# # 5. Custom data-collator that actually pads + masks
# def data_collator(features):
#     # 1) pad inputs & masks
#     batch = tokenizer.pad(
#         features,
#         padding=True,
#         return_tensors="pt",
#     )

#     # 2) labels = all tokens, but mask pad tokens to -100
#     labels = batch["input_ids"].clone()
#     labels[batch["attention_mask"] == 0] = -100

#     batch["labels"] = labels
#     return batch

def data_collator(features):
    # 1) Extract all label lists before padding
    all_labels = [f["labels"] for f in features]

    # 2) Let the tokenizer pad only the input fields
    batch = tokenizer.pad(
        [
            {
                "input_ids":      f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ],
        padding=True,
        return_tensors="pt"
    )

    # 3) Now manually pad each label sequence to match batch["input_ids"].size(1)
    batch_size, seq_len = batch["input_ids"].shape
    padded_labels = []
    for lbl in all_labels:
        # lbl is a Python list of length <= seq_len; pad the remainder with -100
        pad_length = seq_len - len(lbl)
        padded = lbl + [-100] * pad_length
        padded_labels.append(padded)

    # 4) Convert padded_labels into a (batch_size x seq_len) tensor
    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

    return batch

# model.gradient_checkpointing_enable()

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="mistral-gsm8k-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    learning_rate=1e-6,
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
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("mistral-gsm8k-finetuned-final")