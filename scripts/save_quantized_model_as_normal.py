from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch, os

model_name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
save_dir   = "neuralmagic"
os.makedirs(save_dir, exist_ok=True)

# 1. Load quantized model + tokenizer + config
quant_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
config    = AutoConfig.from_pretrained(model_name)

# 2. Strip away any quantization settings so __init__ builds a normal Llama
if hasattr(config, "quantization_config"):
    delattr(config, "quantization_config")
for attr in ("quantization_bits", "bits") :
    if hasattr(config, attr):
        delattr(config, attr)

config.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 3. Extract only your “real” weights 
orig_sd = quant_model.state_dict()
clean_sd = {
    k: v
    for k, v in orig_sd.items()
    if not (k.endswith("weight_scale") or k.endswith("weight_zero_point"))
}

# 4. Build a fresh standard model and load just those tensors
std_model = AutoModelForCausalLM.from_config(config)
std_model.load_state_dict(clean_sd, strict=True)

# 5. Now save it normally
std_model.save_pretrained(save_dir)