import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# Input prompt
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate exactly one token
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=1,  # Generate only one new token
        do_sample=False,   # Greedy decoding for determinism
        pad_token_id=tokenizer.eos_token_id
    )

# Decode only the new token
new_token = output[0][inputs["input_ids"].shape[1]:]
print("Generated token:", tokenizer.decode(new_token))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(layers, noscale_scores, marker='o', label="Original vs NoScale")
plt.plot(layers, hopscotch_scores, marker='o', label="Original vs Hopscotch")
plt.title("MMD Distance from Original across Layers (7-Layer Intervention)", fontsize=18)
plt.xlabel("Layer Number", fontsize=16)
plt.ylabel("MMD", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig(f"MMD.pdf", dpi=300, bbox_inches="tight", format="pdf")