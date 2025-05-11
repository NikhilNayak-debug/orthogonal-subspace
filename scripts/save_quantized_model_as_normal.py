from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import copy
import torch
import torch.nn as nn
from compressed_tensors.linear.compressed_linear import CompressedLinear
from llmcompressor.transformers import SparseAutoModelForCausalLM

save_dir = "plain"

model_name = "/workspace/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config    = AutoConfig.from_pretrained(model_id)

config.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

model = SparseAutoModelForCausalLM.from_pretrained(
  model_name,
  device_map="cpu",
  trust_remote_code=True,
)

def compress_to_plain(model):
    # work on a CPU copy so we can freely pull out float32 weights
    model = copy.deepcopy(model).cpu().eval()

    for name, module in list(model.named_modules()):
        if isinstance(module, CompressedLinear):
            # 1) ask the compressor to decompress into a float32 [out, in] weight
            w_fp: torch.Tensor = module.compressor.decompress_module(module)

            # 2) build a new, standard Linear with identical shape (no bias here)
            linear = nn.Linear(module.in_features, module.out_features, bias=(module.bias is not None))
            linear.weight.data.copy_(w_fp)

            # 3) if there was a bias (rare for GPTQ) copy that too
            if module.bias is not None:
                linear.bias.data.copy_(module.bias.data.cpu())

            # 4) replace in the parent
            parent = model
            *path, attr = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, attr, linear)

    return model

# ——— USAGE ———
# assume `model` is your quantized LlamaForCausalLM
plain = compress_to_plain(model)

# now cast & move back to GPU (matching the rest of your setup)
plain = plain.to(dtype=torch.float)

# sanity check
prompt = "### Question:\nWhat is the capital of France?\n### Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to(plain.device)
with torch.no_grad():
    out = plain.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))

plain.save_pretrained(save_dir)