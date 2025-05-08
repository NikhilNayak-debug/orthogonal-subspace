from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
save_dir   = "neuralmagic"
 
 # 1. Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(
     model_name,
     torch_dtype=torch.bfloat16,
     device_map="auto",
 )
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
 
# 2. Make sure output dir exists & save tokenizer + config
os.makedirs(save_dir, exist_ok=True)
tokenizer.save_pretrained(save_dir)
model.config.save_pretrained(save_dir)
 
# 3. Build a filtered state_dict
clean_state_dict = {
     name: tensor
     for name, tensor in model.state_dict().items()
     if not (name.endswith("weight_scale") or name.endswith("weight_zero_point"))
}
 
# 4. Save the model weights (bfloat16 only) with save_pretrained
model.save_pretrained(save_dir, state_dict=clean_state_dict)