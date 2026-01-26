"""
MR to CT Translation using ControlNet

This script performs MR to CT translation:
1. Encodes MR image to latent space using PatchVolume AE
2. Uses ControlNet to generate CT latent from MR latent
3. Decodes CT latent to CT image using PatchVolume AE
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import torch
import torchio as tio
import argparse
from ddpm.cldm import ControlLDM
from ddpm.BiFlowNet import GaussianDiffusion
from AutoEncoder.model.PatchVolume import patchvolumeAE
import numpy as np
from tqdm import tqdm
import glob

def encode_mr_to_latent(mr_image_path, AE, device):
    """Encode MR image to latent representation."""
    # Load MR image
    mr_img = tio.ScalarImage(mr_image_path)
    mr_data = mr_img.data  # (1, H, W, D)

    # Normalize to [0, 1]
    img_min = mr_data.min()
    img_max = mr_data.max()
    if img_max > img_min:
        mr_data = (mr_data - img_min) / (img_max - img_min)
    else:
        mr_data = torch.zeros_like(mr_data)

    # Convert to [-1, 1]
    mr_data = mr_data * 2 - 1

    # Transpose to (1, D, H, W)
    mr_data = mr_data.transpose(1, 3).transpose(2, 3)
    mr_data = mr_data.type(torch.float32)

    # Encode to latent
    mr_data = mr_data.to(device)
    with torch.no_grad():
        z = AE.patch_encode_sliding(mr_data, patch_size=64, sliding_window=16)

        # Normalize to [-1, 1] range for diffusion model
        codebook_min = AE.codebook.embeddings.min()
        codebook_max = AE.codebook.embeddings.max()
        mr_latent = ((z - codebook_min) / (codebook_max - codebook_min)) * 2.0 - 1.0

    return mr_latent, mr_img.affine, mr_img.shape


def decode_latent_to_ct(ct_latent, AE, device, affine, original_shape):
    """Decode CT latent to CT image."""
    with torch.no_grad():
        # Denormalize latent
        codebook_min = AE.codebook.embeddings.min()
        codebook_max = AE.codebook.embeddings.max()
        ct_latent_denorm = (((ct_latent + 1.0) / 2.0) *
                           (codebook_max - codebook_min)) + codebook_min

        # Decode to CT volume (patch_size=64 for consistency with BiFlowNet)
        ct_volume = AE.decode_sliding(
            ct_latent_denorm,
            quantize=True,
            patch_size=64,
            sliding_window=2,  # Use overlap for smoother results
            compress_ratio=8
        )

        # Convert back to original orientation (D, H, W) -> (H, W, D)
        ct_volume = ct_volume.squeeze(0).squeeze(0).cpu()  # (D, H, W)
        ct_volume = ct_volume.transpose(0, 2).transpose(0, 1)  # (H, W, D)

    return ct_volume


def mr_to_ct_translation(mr_image_path, model, diffusion, AE, device, args):
    """Perform MR to CT translation."""
    print(f"\nProcessing: {mr_image_path}")

    # Step 1: Encode MR to latent
    print("  [1/3] Encoding MR to latent...")
    mr_latent, affine, original_shape = encode_mr_to_latent(mr_image_path, AE, device)
    print(f"    MR latent shape: {mr_latent.shape}")

    # Step 2: Generate CT latent using ControlNet
    print("  [2/3] Generating CT latent using ControlNet...")
    with torch.no_grad():
        # Random noise as starting point
        if args.fixed_seed:
            torch.manual_seed(args.seed)
        z = torch.randn_like(mr_latent)

        # Class label (0 for single-class CT model)
        y = torch.tensor([0], device=device)

        # Resolution conditioning
        res = torch.tensor([1, 1, 1], device=device) / 128.0
        res = res.unsqueeze(0)

        # Sample using ControlNet (MR latent as hint)
        ct_latent = diffusion.p_sample_loop(
            model,
            z,
            y=y,
            hint=mr_latent,  # MR latent as condition
            res=res
        )

    print(f"    Generated CT latent shape: {ct_latent.shape}")

    # Step 3: Decode CT latent to CT volume
    print("  [3/3] Decoding CT latent to volume...")
    ct_volume = decode_latent_to_ct(ct_latent, AE, device, affine, original_shape)
    print(f"    CT volume shape: {ct_volume.shape}")

    return ct_volume, affine


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load AutoEncoder
    print(f"\nLoading AutoEncoder from: {args.AE_ckpt}")
    AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt, map_location=device)
    AE.eval()

    # Load ControlLDM
    print(f"Loading ControlLDM...")
    model = ControlLDM(
        dim=args.model_dim,
        dim_mults=args.dim_mults,
        channels=args.volume_channels,
        init_kernel_size=3,
        cond_classes=args.num_classes,
        learn_sigma=False,
        use_sparse_linear_attn=args.use_attn,
        vq_size=args.vq_size,
        num_mid_DiT=args.num_dit,
        patch_size=args.patch_size,
        control_scales=args.control_scale
    ).to(device)

    # Load pretrained LDM
    print(f"  Loading pretrained LDM from: {args.ldm_ckpt}")
    model.load_pretrained_ldm(args.ldm_ckpt)

    # Load ControlNet weights
    print(f"  Loading ControlNet from: {args.controlnet_ckpt}")
    checkpoint = torch.load(args.controlnet_ckpt, map_location=device)
    model.controlnet.load_state_dict(checkpoint['model'])

    model.eval()

    # Setup diffusion
    diffusion = GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).to(device)

    # Get list of MR images
    if os.path.isdir(args.input_mr):
        mr_images = sorted(glob.glob(os.path.join(args.input_mr, "*.nii.gz")))
    else:
        mr_images = [args.input_mr]

    print(f"\nFound {len(mr_images)} MR image(s) to process")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")

    # Process each MR image
    for mr_path in tqdm(mr_images, desc="Translating MR to CT"):
        # Translate MR to CT
        ct_volume, affine = mr_to_ct_translation(
            mr_path, model, diffusion, AE, device, args
        )

        # Save CT image
        filename = os.path.basename(mr_path).replace('.nii.gz', '_translated_CT.nii.gz')
        output_path = os.path.join(args.output_dir, filename)

        ct_image = tio.ScalarImage(tensor=ct_volume.unsqueeze(0), affine=affine)
        ct_image.save(output_path)
        print(f"  ✓ Saved to: {output_path}")

    print("\n✓ All translations completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MR to CT Translation using ControlNet")

    # Model paths
    parser.add_argument("--AE-ckpt", type=str, required=True,
                       help="Path to PatchVolume AutoEncoder checkpoint")
    parser.add_argument("--ldm-ckpt", type=str, required=True,
                       help="Path to pretrained BiFlowNet checkpoint")
    parser.add_argument("--controlnet-ckpt", type=str, required=True,
                       help="Path to trained ControlNet checkpoint")

    # Input/Output
    parser.add_argument("--input-mr", type=str, required=True,
                       help="Path to MR image or directory of MR images")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save translated CT images")

    # Model architecture (must match training config)
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--loss-type", type=str, default='l1')
    parser.add_argument("--model-dim", type=int, default=72)
    parser.add_argument("--dim-mults", type=list, default=[1,1,2,4,8])
    parser.add_argument("--use-attn", type=list, default=[0,0,0,1,1])
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes (1=CT only)")
    parser.add_argument("--vq-size", type=int, default=64)
    parser.add_argument("--num-dit", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--control-scale", type=float, default=1.0)

    # Generation settings
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--fixed-seed", action="store_true",
                       help="Use fixed seed for consistent results")

    args = parser.parse_args()
    main(args)
