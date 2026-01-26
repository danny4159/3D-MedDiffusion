

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from torch.utils.data import DataLoader
from AutoEncoder.model.PatchVolume import patchvolumeAE 
from dataset.Singleres_dataset import Singleres_dataset
import torch
from os.path import join
import argparse
import torchio as tio
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def generate(args):

    tr_dataset = Singleres_dataset(root_dir=args.data_path,generate_latents = True)

    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AE_ckpt = args.AE_ckpt

    print("Loading checkpoint from CPU...")
    AE = patchvolumeAE.load_from_checkpoint(AE_ckpt, map_location='cpu')

    # Remove unnecessary components to save GPU memory
    # During inference, we only need encoder, decoder, and codebook
    print("Removing unnecessary components...")
    if hasattr(AE, 'loss_volume_disc'):
        del AE.loss_volume_disc  # Remove discriminator
        print("  - Removed discriminator")
    if hasattr(AE, 'perceptual_loss'):
        del AE.perceptual_loss  # Remove perceptual loss network
        print("  - Removed perceptual_loss")
    if hasattr(AE, 'perceptual_model'):
        del AE.perceptual_model
        print("  - Removed perceptual_model")

    # Clear cache before loading to GPU
    torch.cuda.empty_cache()
    print(f"GPU memory before loading model: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print("Moving model to GPU...")
    AE = AE.to(device)
    print(f"GPU memory after loading model (FP32): {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    AE.eval()

    # Keep FP32 to avoid NaN issues in encoder
    print(f"Model ready (FP32)!\n")


    for idx, (sample, paths) in enumerate(tr_dataloader):
        print(f"\n=== Processing batch {idx} ===")
        print(f"Input shape: {sample.shape}")
        print(f"GPU memory before: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        sample = sample.cuda()  # Keep FP32
        print(f"GPU memory after moving to GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        with torch.no_grad():
            # Calculate number of patches
            depth_patches = sample.shape[2] // 64
            total_patches = depth_patches * 8 * 8  # (depth/64) * (512/64) * (512/64)

            # Always use sliding window to avoid OOM (process 16 patches at a time)
            print(f"Processing {total_patches} patches using sliding window (batch=16)...")
            z = AE.patch_encode_sliding(sample, patch_size=64, sliding_window=16)

            print(f"Latent shape after patch_encode: {z.shape}")
            print(f"  z min: {z.min():.6f}, max: {z.max():.6f}, mean: {z.mean():.6f}")
            print(f"GPU memory after encoding: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

            # Get codebook range for normalization
            codebook_min = AE.codebook.embeddings.min()
            codebook_max = AE.codebook.embeddings.max()
            print(f"  codebook_min: {codebook_min:.6f}, codebook_max: {codebook_max:.6f}")

            output = ((z - codebook_min) /
            (codebook_max - codebook_min)) * 2.0 - 1.0
            print(f"Output shape after normalization: {output.shape}")
            print(f"  output min: {output.min():.6f}, max: {output.max():.6f}, mean: {output.mean():.6f}")

        output = output.float().cpu()  # Convert back to FP32 for saving
        print(f"GPU memory after moving to CPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # Clear cache after each batch
        torch.cuda.empty_cache()
        for i, path in enumerate(paths):
            output_ = output[i]
            print(f"  Before save - output_[{i}] shape: {output_.shape}, min: {output_.min():.6f}, max: {output_.max():.6f}, mean: {output_.mean():.6f}")

            dir_name = os.path.basename(os.path.dirname(path))
            latent_dir_name = dir_name + '_latents'
            path = path.replace(dir_name, latent_dir_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img = tio.ScalarImage(tensor = output_ )
            img.save(path)
            print(f"  Saved to: {path}")   

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--AE-ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2) 
    parser.add_argument("--num-workers", type=int, default=8) 
    args = parser.parse_args()
    generate(args)



