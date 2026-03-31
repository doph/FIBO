
import torch
import numpy as np
from diffusers import AutoencoderKLWan
import os

def check_stats():
    print("Loading VAE...")
    # Load original VAE to check config
    model_path = "/home/ckenny/models/FIBO/vae_bak/" # Assuming this exists, or use finetuned
    if not os.path.exists(model_path):
        model_path = "/home/ckenny/code/FIBO/outputs/models_v4_bf16/checkpoint-500/vae"
    
    try:
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
    except:
        vae = AutoencoderKLWan.from_pretrained(model_path)
        
    vae.to("cuda", dtype=torch.bfloat16)
    vae.eval()
    
    print(f"\n--- Config Stats ---")
    if hasattr(vae.config, "latents_mean"):
        l_mean = torch.tensor(vae.config.latents_mean)
        l_std = torch.tensor(vae.config.latents_std)
        print(f"Config Mean (avg): {l_mean.mean().item():.4f}")
        print(f"Config Std (avg): {l_std.mean().item():.4f}")
        print(f"Config Mean (min/max): {l_mean.min().item():.4f}, {l_mean.max().item():.4f}")
        print(f"Config Std (min/max): {l_std.min().item():.4f}, {l_std.max().item():.4f}")
    else:
        print("No latents_mean/std in config.")
        
    print(f"\n--- Encoder Output Stats ---")
    # Create random image
    img = torch.randn(1, 3, 256, 256).to("cuda", dtype=torch.bfloat16)
    img = img.unsqueeze(2)
    
    with torch.no_grad():
        posterior = vae.encode(img).latent_dist
        z = posterior.sample()
        
    print(f"Encoded z shape: {z.shape}")
    print(f"z mean: {z.mean().item():.4f}")
    print(f"z std: {z.std().item():.4f}")
    print(f"z min/max: {z.min().item():.4f}, {z.max().item():.4f}")

if __name__ == "__main__":
    check_stats()
