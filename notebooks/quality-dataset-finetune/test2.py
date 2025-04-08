# Configurable parameters:
OUTPUT_MODEL_NAME = "llama_svd_quality"         # Name for the saved model after fine-tuning.

import os
import json
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np
import os
# Import additional modules for distributed training and FSDP.
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.autograd.set_detect_anomaly(True)

###################################################
# 1. Helper Functions for SVD and Parameter Management
###################################################

def decompose_weight_matrix(weight: torch.Tensor, top_k: int):
    """
    Perform SVD on a 2D weight matrix and split into:
      - top_k singular vectors (treated as frozen/buffers)
      - the rest (treated as trainable)
    Returns a dictionary containing:
      {
        "U_high": ...  # buffer
        "S_high": ...  # buffer
        "V_high": ...  # buffer
        "U_low": ...   # parameter
        "S_low": ...   # parameter
        "V_low": ...   # parameter
        "rank_high": top_k
      }
    """
    device_local = weight.device
    # W = weight.to(torch.float32)  # ensure float32 for SVD
    W = weight.detach().cpu().to(torch.float32)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    # Ensure we don’t ask for more than available
    k = min(top_k, S.shape[0])

    # High subspace (frozen)
    U_high = U[:, :k].detach().to(dtype=torch.bfloat16, device=device_local)
    S_high = S[:k].detach().to(dtype=torch.bfloat16, device=device_local)
    V_high = Vt[:k, :].detach().to(dtype=torch.bfloat16, device=device_local)

    # Low subspace (trainable)
    U_low = U[:, k:].detach().to(dtype=torch.bfloat16, device=device_local)
    S_low = S[k:].detach().to(dtype=torch.bfloat16, device=device_local)
    V_low = Vt[k:, :].detach().to(dtype=torch.bfloat16, device=device_local)

    return {
        "U_high": U_high,
        "S_high": S_high,
        "V_high": V_high,
        "U_low": nn.Parameter(U_low),
        "S_low": nn.Parameter(S_low),
        "V_low": nn.Parameter(V_low),
        "rank_high": k
    }


def reconstruct_weight_matrix(svd_dict):
    """
    Reconstruct the full weight matrix:
       W = U_high * diag(S_high) * V_high^T + U_low * diag(S_low) * V_low^T
    """
    U_high = svd_dict["U_high"]
    S_high = svd_dict["S_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    S_low = svd_dict["S_low"]
    V_low = svd_dict["V_low"]

    if U_high.shape[1] > 0 and S_high.shape[0] > 0:
        high_part = torch.mm(U_high * S_high.unsqueeze(0), V_high)
    else:
        high_part = torch.zeros(U_low.size(0), V_low.size(1), device=U_high.device)

    if U_low.shape[1] > 0 and S_low.shape[0] > 0:
        US_low = U_low * S_low.unsqueeze(0)
        low_part = torch.mm(US_low, V_low)
    else:
        low_part = torch.zeros(U_high.size(0), V_high.size(1), device=U_low.device)

    return high_part + low_part


def check_reconstruction_error(weight, svd_dict, atol=1e-5):
    # Move the weight to the same device as the U_high buffer
    target_device = svd_dict["U_high"].device
    weight = weight.to(target_device)
    W_recon = reconstruct_weight_matrix(svd_dict)
    # Ensure reconstruction is also on the target device
    W_recon = W_recon.to(target_device)
    error = torch.norm(weight - W_recon) / torch.norm(weight)
    if error > atol:
        print(f"Warning: Reconstruction error {error:.2e} exceeds tolerance {atol}")
    return error


def project_gradient_to_orthogonal_space(svd_dict):
    """
    Remove from the gradients of the low subspace any component that lies
    in the high subspace.
    """
    if (svd_dict["U_low"].grad is None and
        svd_dict["S_low"].grad is None and
        svd_dict["V_low"].grad is None):
        return

    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]
    U_low  = svd_dict["U_low"]
    V_low  = svd_dict["V_low"]

    if svd_dict["U_low"].grad is not None:
        dU = svd_dict["U_low"].grad
        if dU.dim() == 1:
            dU = dU.view(U_high.shape[0], -1)  # reshape to (m, rank_low) 
        # Now do the projection
        U_high = U_high.to(dU.device)
        proj = U_high @ (U_high.transpose(0,1) @ dU)
        dU.sub_(proj)
        # Flatten again if FSDP expects it flattened
        U_low.grad = dU.view(-1)

    if svd_dict["V_low"].grad is not None:
        dV = svd_dict["V_low"].grad
        if dV.dim() == 1:
            dV = dV.view(-1, V_high.shape[1])
        # Now do the projection
        V_high = V_high.to(dV.device)
        proj = (dV @ V_high.transpose(0,1)) @ V_high
        dV.sub_(proj)
        # Flatten again if FSDP expects it flattened
        V_low.grad = dV.view(-1)
    # We leave S_low unchanged


def compute_effective_rank(matrix):
    """
    Compute the effective rank of a matrix based on the definition provided.
    """
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    singular_values = S.cpu().numpy()

    # Compute the singular value distribution (p_k)
    l1_norm = np.sum(np.abs(singular_values))
    p_k = singular_values / l1_norm

    # Compute the Shannon entropy
    H = -np.sum(p_k * np.log(p_k + 1e-10))  # Add a small constant to avoid log(0)

    # Compute the effective rank
    effective_rank = np.exp(H)

    return effective_rank


###################################################
# 2. LLaMA Model Subclass with SVD (Only for Selected Parameters)
###################################################

class LlamaWithSVD(LlamaForCausalLM):
    """
    Subclass that, on initialization, decomposes selected weight matrices via SVD.
    Only parameters specified in the svd_config are decomposed.
    For each such 2D weight, we freeze the top singular vectors (50% by default)
    and register the lower half (trainable) as parameters.

    Additionally, we pre-compute the module mapping for faster weight injection.
    """
    def __init__(self, config: LlamaConfig, svd_config=None, initialize_svd=True):
        super().__init__(config)
        # svd_config is a dict mapping full parameter names to top_k values.
        self.svd_config = svd_config if svd_config is not None else {}
        self.name_mapping = {}         # maps original name -> safe name
        self.svd_original_mapping = {} # maps safe name -> original name
        self.svd_params = nn.ModuleDict()
        self.svd_module_mapping = {}   # maps safe name -> (module, attribute_name)
        if initialize_svd:
          self._initialize_svd_parameters()

    def reinitialize_svd(self):
        """
        Reinitialize the SVD decomposition on the current (loaded) weights.
        Before reinitialization, store a copy of the original weights for each target parameter,
        then after reinitialization, check and print the reconstruction error.
        """
        # Save original weights for each parameter to be decomposed.
        self._original_weights = {}
        for orig_name in self.svd_config.keys():
            # Retrieve from the model's state_dict; ensure it is on the correct device.
            self._original_weights[orig_name] = self.state_dict()[orig_name].clone().to(device)

        # Clear previous SVD mappings.
        self.name_mapping = {}
        self.svd_original_mapping = {}
        self.svd_params = nn.ModuleDict()
        self.svd_module_mapping = {}
        # Reinitialize the SVD decomposition using the current weights.
        self._initialize_svd_parameters()

        # Now, for each decomposed parameter, compute and print the reconstruction error.
        for orig_name, safe_name in self.name_mapping.items():
            orig_weight = self._original_weights[orig_name]
            svd_dict = {
                "U_high": getattr(self, f"{safe_name}_U_high"),
                "S_high": getattr(self, f"{safe_name}_S_high"),
                "V_high": getattr(self, f"{safe_name}_V_high"),
                "U_low": self.svd_params[safe_name].U_low,
                "S_low": self.svd_params[safe_name].S_low,
                "V_low": self.svd_params[safe_name].V_low
            }
            error = check_reconstruction_error(orig_weight, svd_dict)
            print(f"Reconstruction error for {orig_name}: {error:.2e}")

    def _initialize_svd_parameters(self):
        # Iterate over all parameters
        for name, param in list(self.named_parameters()):
            if len(param.shape) == 2 and name in self.svd_config and self.svd_config[name] > 0:
                top_k = self.svd_config[name]
                print(f"[SVD Init] Decomposing {name} with top_k={top_k}")
                svd_dict = decompose_weight_matrix(param.data, top_k=top_k)
                safe_name = name.replace(".", "_")
                self.name_mapping[name] = safe_name
                self.svd_original_mapping[safe_name] = name

                # Compute the residual: the difference between the original weight and its SVD reconstruction.
                # residual = (param.data - reconstruct_weight_matrix(svd_dict)).detach()
                # Register the residual as a buffer (no gradients).
                # self.register_buffer(f"{safe_name}_residual", residual)

                # Register buffers for the high subspace
                self.register_buffer(f"{safe_name}_U_high", svd_dict["U_high"])
                self.register_buffer(f"{safe_name}_S_high", svd_dict["S_high"])
                self.register_buffer(f"{safe_name}_V_high", svd_dict["V_high"])

                # Create a module to hold the low subspace trainable parameters
                module_svd = nn.Module()
                module_svd.U_low = nn.Parameter(svd_dict["U_low"])
                module_svd.S_low = nn.Parameter(svd_dict["S_low"])
                module_svd.V_low = nn.Parameter(svd_dict["V_low"])
                module_svd.rank_high = svd_dict["rank_high"]
                module_svd.safe_name = safe_name
                self.svd_params[safe_name] = module_svd

                # Freeze the original parameter
                param.requires_grad = False

                # Pre-compute and store the module and attribute name for quick access
                mod, attr = self._get_module_by_name(name)
                if mod is not None:
                    self.svd_module_mapping[safe_name] = (mod, attr)
            # For parameters not in svd_config, leave them trainable (do nothing)

    def _reconstruct_weight(self, original_name):
        safe_name = self.name_mapping[original_name]
        U_high = getattr(self, f"{safe_name}_U_high")
        S_high = getattr(self, f"{safe_name}_S_high")
        V_high = getattr(self, f"{safe_name}_V_high")
        module_svd = self.svd_params[safe_name]
        U_low = module_svd.U_low
        S_low = module_svd.S_low
        V_low = module_svd.V_low
        svd_dict = {"U_high": U_high, "S_high": S_high, "V_high": V_high,
                    "U_low": U_low, "S_low": S_low, "V_low": V_low}
        W = reconstruct_weight_matrix(svd_dict)

        # Retrieve the residual that was stored during initialization.
        # residual = getattr(self, f"{safe_name}_residual").detach()

        # return W + residual

        return W

    def forward(self, *args, **kwargs):
        # Instead of recomputing the module mapping each time,
        # iterate over the precomputed svd_module_mapping.
        for safe_name, (module, attr) in self.svd_module_mapping.items():
            original_name = self.svd_original_mapping[safe_name]
            W = self._reconstruct_weight(original_name)
            # if attr in module._parameters:
            #     print(type(module._parameters))
            #     print(module._parameters)
            #     print(attr)
            #     module._parameters.pop(attr)
            # setattr(module, attr, W)
            # print(module._parameters)
            target_device = getattr(module, attr).device
            W = W.to(target_device)
            with torch.no_grad():
                getattr(module, attr).data.copy_(W)
        return super().forward(*args, **kwargs)

    def _get_module_by_name(self, name):
        """
        Given a full parameter name (e.g. "encoder.block.0.layer.0.SelfAttention.q.weight"),
        return (module, attribute_name) where module.attribute_name is that parameter.
        """
        parts = name.split(".")
        attr = parts[-1]
        mod = self
        for p in parts[:-1]:
            if hasattr(mod, p):
                mod = getattr(mod, p)
            elif p.isdigit():
                mod = mod[int(p)]
            else:
                return None, None
        return mod, attr

    def project_gradients(self):
        for safe_name, module_svd in self.svd_params.items():
            svd_dict = {
                "U_high": getattr(self, f"{safe_name}_U_high"),
                "S_high": getattr(self, f"{safe_name}_S_high"),
                "V_high": getattr(self, f"{safe_name}_V_high"),
                "U_low": module_svd.U_low,
                "S_low": module_svd.S_low,
                "V_low": module_svd.V_low,
            }
            project_gradient_to_orthogonal_space(svd_dict)

###################################################
# 3. Utility: Auto-generate SVD Config for Target Parameters
###################################################
def auto_generate_target_svd_config(model):
    """
    Given a model, generate an SVD configuration dictionary only for parameters that contain one of the
    following substrings:
      - self_attn.q_proj
      - self_attn.k_proj
      - self_attn.v_proj
      - self_attn.o_proj
      - mlp.gate_proj
      - mlp.down_proj
      - mlp.up_proj
    For each such 2D parameter, set:
         top_k = floor(min(dim0, dim1) / 2)
    """
    target_patterns = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj"
    ]
    config = {}
    for name, param in model.named_parameters():
        if any(pat in name for pat in target_patterns) and len(param.shape) == 2:
            # effective_rank = compute_effective_rank(param.data)
            # top_k = int(np.floor(effective_rank))
            # full_rank = min(param.shape)
            # if top_k > full_rank:
            #     top_k = full_rank
            # config[name] = top_k
            top_k = int(np.floor(max(param.shape)*0.50))
            full_rank = min(param.shape)
            if top_k > full_rank:
                top_k = full_rank
            config[name] = top_k
    # save_svd_config(config)
    return config

# Define Dataset
class QualityDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data["messages"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        return messages

# Collate function with reduced max_length (optional)
def collate_fn(batch, tokenizer, max_length=2048):
    prompts = [
        tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        for sample in batch
    ]
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)

    labels = encodings["input_ids"].clone()

    # Mask only special tokens (e.g., <|endoftext|>, etc.)
    for special_token_id in tokenizer.all_special_ids:
        if special_token_id == 128009:
            continue
        labels[labels == special_token_id] = -100
    
    special_token_ids = [
        tokenizer.convert_tokens_to_ids(t)
        for t in ["<|start_header_id|>", "<|end_header_id|>"]
        if tokenizer.convert_tokens_to_ids(t) is not None
    ]
    
    for token_id in special_token_ids:
        labels[labels == token_id] = -100

    encodings["labels"] = labels
    return encodings

###################################################

# 5. Training and Saving the SVD Model on Amazon Reviews
###################################################
def train_svd_model(output_model_name=OUTPUT_MODEL_NAME, device=None, local_rank=None):

    train_path = "/new_data/knowledge_rh/quality/training_mix/train_base_extractive_stack.jsonl"

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ✅ Add a new pad token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = LlamaConfig.from_pretrained(model_name)
    config.use_cache = False  # if applicable for LLaMA; otherwise remove or adjust
    config.attention_dropout = 0.2
    config.hidden_dropout = 0.2

    # Create datasets and dataloaders
    train_dataset = QualityDataset(train_path, tokenizer)
    sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler,
                            collate_fn=lambda b: collate_fn(b, tokenizer))
    
    # (Optional) Print one batch for debugging on rank 0.
    if dist.get_rank() == 0:
        for batch in train_loader:
            print("Input IDs:", tokenizer.decode(batch['input_ids'][0]))
            print("Labels:", tokenizer.decode([x for x in batch['labels'][0] if x != -100]))
            # print(batch['input_ids'][0])
            # print(batch['labels'][0])
            break

    # Load a standard LLaMA model to generate the SVD config.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config
    )
    base_model = base_model.to(device)
    target_svd_config = auto_generate_target_svd_config(base_model)
    if dist.get_rank() == 0:
        print("Auto-generated target SVD config:")
        for k, v in target_svd_config.items():
            print(f"  {k}: freeze top {v} singular vectors")

    del base_model
    torch.cuda.empty_cache()

    # Initialize our custom SVD model with target_svd_config.
    model = LlamaWithSVD.from_pretrained(model_name, config=config, svd_config=target_svd_config, initialize_svd=False, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    # Ensure pad_token_id is correctly set
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move the model to the local device.
    # model = model.to(device)

    model.reinitialize_svd()
    # model.gradient_checkpointing_enable()

    # ----------------------------
    # FSDP Wrapping:
    # Wrap the model with FSDP after moving it to the correct device.
    # ----------------------------
    model = FSDP(model, use_orig_params=True, cpu_offload=CPUOffload(offload_params=True), device_id=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
    num_epochs = 5  # adjust as needed

    model.train()
    for epoch in range(num_epochs):
        # Reset the sampler epoch for shuffling.
        sampler.set_epoch(epoch)
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True)
        start_time = time.time()

        for batch in progress_bar:
            # Move batch to the current device.
            for key, val in batch.items():
                batch[key] = val.to(device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            model.project_gradients()  # ensure gradients remain in correct subspace
            optimizer.step()
            
            # (Optional) Log loss only on rank 0.
            if dist.get_rank() == 0:
                with open("loss.txt", "a") as f:  # "a" mode appends to the file
                    print(f"Loss: {loss}", file=f)

            total_loss += loss.item()
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (progress_bar.n + 1) * (len(train_loader) - progress_bar.n)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", eta=f"{remaining_time:.2f}s")

        avg_loss = total_loss / len(train_loader)
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

            # Save model checkpoint after each epoch only on rank 0.
            epoch_model_path = f"{output_model_name}_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), epoch_model_path)
            print(f"Model checkpoint saved: {epoch_model_path}")

    return model, tokenizer, train_dataset, None

# ----------------------------
# 6. Main: Distributed Initialization and Cleanup
# ----------------------------
if __name__ == "__main__":
    # Initialize distributed process group for FSDP.
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Run the training.
    model, tokenizer, train_dataset, _ = train_svd_model(output_model_name=OUTPUT_MODEL_NAME, device=device, local_rank=local_rank)

    # (Optional) Clean up the process group.
    dist.destroy_process_group()