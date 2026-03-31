
import torch
from diffusers import AutoencoderKLWan
import argparse

def main():
    print("Loading model...")
    try:
        vae = AutoencoderKLWan.from_pretrained(
            "/home/ckenny/models/FIBO/vae/", subfolder="vae", torch_dtype=torch.float32
        )
    except:
        vae = AutoencoderKLWan.from_pretrained(
            "/home/ckenny/models/FIBO/vae/", torch_dtype=torch.float32
        )

    print("\n--- Top Level Modules ---")
    for name, module in vae.named_children():
        print(f"Module: {name}")

    # Simulate the freezing logic from the script
    print("\n--- Simulating Freezing Logic ---")
    # Script logic:
    # if args.train_only_decoder:
    #    for param in vae.encoder.parameters():
    #        param.requires_grad = False
    
    # Apply it
    if hasattr(vae, "encoder"):
        for param in vae.encoder.parameters():
            param.requires_grad = False
        print("Froze vae.encoder")
    else:
        print("vae.encoder not found!")

    print("\n--- Checking Trainable Parameters ---")
    trainable_params = []
    for name, param in vae.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    
    if not trainable_params:
        print("No trainable parameters.")
    else:
        print(f"Found {len(trainable_params)} trainable parameters.")
        print("First 10 trainable parameters:")
        for p in trainable_params[:10]:
            print(f" - {p}")
        
        # Check specifically for quant_conv
        has_quant_conv = any("quant_conv" in p for p in trainable_params)
        if has_quant_conv:
            print("\nCRITICAL: 'quant_conv' parameters are TRAINABLE!")
        else:
            print("\n'quant_conv' parameters are frozen.")

if __name__ == "__main__":
    main()
