#!/usr/bin/env python3
"""
Minimal patch script: apply the "SeamlessWrapper" to FFN/MLP modules
but load the FunctionGemma checkpoint (google/functiongemma-270m-it).

Usage (Colab / local):
  - Ensure HF_TOKEN set in env or Colab secrets.
  - python scripts/seamless_functiongemma_patch.py --save-dir ./SEAMLESS-FUNCTIONGEMMA-270M
"""
import argparse
import os
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def log(msg):
    print(f"[LOG]: {msg}")

def wrap_tensor(inputs: torch.Tensor) -> torch.Tensor:
    wrapped = torch.cat([inputs[:, :, -1:], inputs, inputs[:, :, :1]], dim=2)
    wrapped = torch.cat([wrapped[:, -1:], wrapped, wrapped[:, :1]], dim=1)
    return wrapped

class SeamlessWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if x.dim() != 3:
            return self.module(x)
        batch_size, seq_len, hidden_dim = x.shape
        sqrt_val = math.isqrt(seq_len)
        if sqrt_val ** 2 != seq_len:
            return self.module(x)
        try:
            x_processed = self.module(x)
            x_reshaped = x_processed.contiguous().view(batch_size, sqrt_val, sqrt_val, hidden_dim)
            x_wrapped = wrap_tensor(x_reshaped)
            new_side = sqrt_val + 2
            new_seq_len = new_side ** 2
            x_final = x_wrapped.view(batch_size, new_seq_len, hidden_dim)
            x_sliced = x_final[:, :seq_len, :].contiguous()
            return x_sliced
        except Exception as e:
            log(f"Wrap failed: {e}")
            return x_processed

def wrap_ffn(module):
    for name, child in list(module.named_children()):
        if 'mlp' in name.lower() or 'ffn' in name.lower():
            try:
                setattr(module, name, SeamlessWrapper(child))
                log(f"Wrapped '{name}'")
            except Exception as e:
                log(f"Failed to wrap '{name}': {e}")
        else:
            wrap_ffn(child)

def main(args):
    model_name = args.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device)
    wrap_ffn(model)
    os.makedirs(args.save_dir, exist_ok=True)
    log("Moving model to CPU and saving state_dict (tokenizer saved too).")
    model.to("cpu")
    torch.save(model.state_dict(), os.path.join(args.save_dir, "model_state_dict.pth"))
    tokenizer.save_pretrained(args.save_dir)
    log(f"Saved patched model to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/functiongemma-270m-it")
    parser.add_argument("--save-dir", default="./SEAMLESS-FUNCTIONGEMMA-270M")
    main(parser.parse_args())
