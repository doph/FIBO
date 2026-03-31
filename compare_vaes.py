
import torch
import numpy as np
from diffusers import AutoencoderKLWan
import os

def compare_vaes():
    print("Loading VAEs...")
    orig_path = "/home/ckenny/models/FIBO/vae_bak/"
    ft_path = "/home/ckenny/code/FIBO/outputs/models_v4_bf16/checkpoint-500/vae"
    
    if not os.path.exists(ft_path):
        ft_path = "/home/ckenny/code/FIBO/outputs/models_v4_bf16/checkpoint-500"

    try:
        vae_orig = AutoencoderKLWan.from_pretrained(orig_path, subfolder="vae")
    except:
        vae_orig = AutoencoderKLWan.from_pretrained(orig_path)
        
    try:
        vae_ft = AutoencoderKLWan.from_pretrained(ft_path, subfolder="vae")
    except:
        vae_ft = AutoencoderKLWan.from_pretrained(ft_path)
        
    vae_orig.eval().to("cuda", dtype=torch.bfloat16)
    vae_ft.eval().to("cuda", dtype=torch.bfloat16)
    
    print("\n--- Post Quant Conv Comparison ---")
    # Check if weights changed
    if hasattr(vae_orig, "post_quant_conv") and hasattr(vae_ft, "post_quant_conv"):
        pqc_orig = vae_orig.post_quant_conv.weight
        pqc_ft = vae_ft.post_quant_conv.weight
        
        diff = (pqc_orig - pqc_ft).abs().max()
        print(f"Post Quant Conv Max Diff: {diff.item()}")
        
        if diff > 0:
            print("Post Quant Conv weights CHANGED.")
            print(f"Orig Mean/Std: {pqc_orig.mean().item():.4f} / {pqc_orig.std().item():.4f}")
            print(f"FT Mean/Std:   {pqc_ft.mean().item():.4f} / {pqc_ft.std().item():.4f}")
        else:
            print("Post Quant Conv weights UNCHANGED.")
    else:
        print("post_quant_conv not found.")

    print("\n--- Decoder Comparison ---")
    # Decoder was trained, should differ
    z = torch.randn(1, 48, 1, 16, 16).to("cuda", dtype=torch.bfloat16)
    
    with torch.no_grad():
        dec_orig = vae_orig.decode(z).sample
        dec_ft = vae_ft.decode(z).sample
        
    diff_dec = (dec_orig - dec_ft).abs().mean()
    print(f"Decoder Output Avg Diff: {diff_dec.item()}")
    
    # Check if FT output is broken (e.g. all zeros or high variance)
    print(f"Orig Output Mean/Std: {dec_orig.mean().item():.4f} / {dec_orig.std().item():.4f}")
    print(f"FT Output Mean/Std:   {dec_ft.mean().item():.4f} / {dec_ft.std().item():.4f}")

if __name__ == "__main__":
    compare_vaes()
