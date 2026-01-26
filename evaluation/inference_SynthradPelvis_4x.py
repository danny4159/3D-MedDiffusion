import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
from ddpm.BiFlowNet import GaussianDiffusion
from ddpm import BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio
import numpy as np
from torch.cuda.amp import autocast

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = BiFlowNet(
        dim=args.model_dim,
        dim_mults=args.dim_mults,
        channels=args.volume_channels,
        init_kernel_size=3,
        cond_classes=args.num_classes,
        learn_sigma=False,
        use_sparse_linear_attn=args.use_attn,
        vq_size=args.vq_size,
        num_mid_DiT=args.num_dit,
        patch_size=args.patch_size
    ).cuda()

    diffusion = GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).cuda()

    model_ckpt = torch.load(args.model_ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(model_ckpt['ema'], strict=True)
    model = model.cuda()
    model.eval()

    AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt, map_location='cpu').cuda()
    AE.eval()

    device = torch.device("cuda")
    spacing = tuple(args.spacing)  # (H_spacing, W_spacing, D_spacing)
    affine = np.diag(spacing + (1,))

    print(f"Generating {args.num_samples} samples...")
    print(f"Resolution: {args.resolution}")
    print(f"Output dir: {args.output_dir}")

    # Generate samples
    for i in range(args.num_samples):
        print(f"Sample {i+1}/{args.num_samples}...", end=" ", flush=True)
        with torch.no_grad():
            with autocast(enabled=args.enable_amp):
                z = torch.randn(1, args.volume_channels, args.resolution[0], args.resolution[1], args.resolution[2], device=device)
                y = torch.tensor([0], device=device)  # Class 0 (single class for SynthradPelvis)
                res_emb = torch.tensor(args.resolution, device=device).float() / 64.0

                samples = diffusion.p_sample_loop(
                    model, z, y=y, res=res_emb
                )

                samples = (((samples + 1.0) / 2.0) * (AE.codebook.embeddings.max() -
                                                    AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()

                # Decode
                volume = AE.decode_sliding(samples, quantize=True, patch_size=64, sliding_window=4, compress_ratio=args.compress_ratio)

            volume_path = os.path.join(args.output_dir, f'sample_{i:03d}.nii.gz')
            volume = volume.detach().squeeze(0).cpu()
            volume = volume.transpose(1, 3).transpose(1, 2)
            tio.ScalarImage(tensor=volume, affine=affine).save(volume_path)
            print(f"Saved to {volume_path}")

        torch.cuda.empty_cache()

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--AE-ckpt", type=str, required=True, help="Path to Autoencoder Checkpoint")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to BiFlowNet Checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save generated images")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--resolution", type=int, nargs=3, default=[32, 96, 80], help="Latent resolution (D, H, W)")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Voxel spacing (H, W, D) in mm")
    parser.add_argument("--compress-ratio", type=int, default=4, help="Autoencoder compression ratio (4 for 4x AE)")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model-dim", type=int, default=72)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--num-dit", type=int, default=1)
    parser.add_argument("--dim-mults", type=list, default=[1, 1, 2, 4, 8])
    parser.add_argument("--use-attn", type=list, default=[0, 0, 0, 1, 1])
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--loss-type", type=str, default='l1')
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--vq-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-amp", type=bool, default=True, help="Enable Automatic Mixed Precision")
    args = parser.parse_args()
    main(args)
