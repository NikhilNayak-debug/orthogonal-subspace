from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
import torch.nn as nn
from awq.utils.packing_utils import dequantize_gemm

model = AutoAWQForCausalLM.from_quantized(
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    trust_remote_code=True
)

def convert_awq_to_fp16(model):
    for name, module in model.named_modules():
        # Identify AWQ quantized linear layers by their attributes
        if hasattr(module, "qweight") and hasattr(module, "scales") and hasattr(module, "qzeros"):
            # Dequantize the INT4 weights to FP32
            fp32_w = dequantize_gemm(
                module.qweight, module.qzeros, module.scales,
                bits=4, group_size=module.group_size
            ).T
            # Create a new FP16 nn.Linear with the same dims
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name) if parent_name else model
            linear_fp16 = nn.Linear(
                module.in_features, module.out_features,
                bias=(module.bias is not None)
            ).to(torch.float16).to(fp32_w.device)
            linear_fp16.weight.data = fp32_w.to(torch.float16)
            if module.bias is not None:
                linear_fp16.bias.data = module.bias.to(torch.float16)
            # Replace the quantized module
            setattr(parent, attr, linear_fp16)
    return model

model = convert_awq_to_fp16(model)

# ─── 2. Load the _plain_ LLaMA-3.1-8B-Instruct config & tokenizer ─── #
plain_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
config    = LlamaConfig.from_pretrained(plain_model_id)
tokenizer = AutoTokenizer.from_pretrained(plain_model_id)

# ─── 3. Instantiate a fresh plain model and copy weights ─── #
plain_model = LlamaForCausalLM(config)

# Copy over every parameter from your dequantized model
plain_model.load_state_dict(model.model.state_dict())

# ─── 4. Tie & cast to match HF defaults ─── #
plain_model.tie_weights()

prompt = "### Question:\nWhat is the capital of France?\n### Answer:"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = plain_model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))

# ─── 5. Save out a new “plain” LLaMA-Instruct folder ─── #
out_dir = "plain_awq"
plain_model.save_pretrained(out_dir, safe_serialization=True)
tokenizer.save_pretrained(out_dir)
config.save_pretrained(out_dir)

print(f"✅ Saved plain LLaMA-3.1-8B-Instruct to '{out_dir}'")