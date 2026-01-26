import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "."))
sys.path.append(project_root)
from torch.utils.data import DataLoader
from AutoEncoder.model.PatchVolume import patchvolumeAE
from dataset.Singleres_dataset import Singleres_dataset
import torch
import torchio as tio
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

# Test with single sample
tr_dataset = Singleres_dataset(root_dir='config/SingleRes_dataset.json', generate_latents=True)
tr_dataloader = DataLoader(tr_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AE_ckpt = 'results/PatchVolume_8x_stage2/my_model/version_9/checkpoints/latest_checkpoint.ckpt'

print("Loading model...")
AE = patchvolumeAE.load_from_checkpoint(AE_ckpt, map_location='cpu')

# Remove unnecessary components
if hasattr(AE, 'perceptual_loss'):
    del AE.perceptual_loss

torch.cuda.empty_cache()
AE = AE.to(device)
AE.eval()

print("Model loaded (FP32)!\n")

# Process only first sample
for sample, paths in tr_dataloader:
    print(f"Input shape: {sample.shape}")
    print(f"Input min: {sample.min():.6f}, max: {sample.max():.6f}, mean: {sample.mean():.6f}")

    sample = sample.cuda()

    with torch.no_grad():
        # Use sliding window to avoid OOM
        print("Using sliding window encoding (batch=16)...")
        z = AE.patch_encode_sliding(sample, patch_size=64, sliding_window=16)
        print(f"\nLatent z shape: {z.shape}")
        print(f"z min: {z.min():.6f}, max: {z.max():.6f}, mean: {z.mean():.6f}, std: {z.std():.6f}")

        codebook_min = AE.codebook.embeddings.min()
        codebook_max = AE.codebook.embeddings.max()
        print(f"codebook_min: {codebook_min:.6f}, codebook_max: {codebook_max:.6f}")

        output = ((z - codebook_min) / (codebook_max - codebook_min)) * 2.0 - 1.0
        print(f"\nOutput shape: {output.shape}")
        print(f"output min: {output.min():.6f}, max: {output.max():.6f}, mean: {output.mean():.6f}, std: {output.std():.6f}")

    output = output.float().cpu()
    print(f"\nAfter CPU conversion:")
    print(f"output min: {output.min():.6f}, max: {output.max():.6f}, mean: {output.mean():.6f}, std: {output.std():.6f}")

    # Save
    output_ = output[0]
    print(f"\nBefore save - output_[0] shape: {output_.shape}")
    print(f"output_[0] min: {output_.min():.6f}, max: {output_.max():.6f}, mean: {output_.mean():.6f}")

    test_path = 'test_latent.nii.gz'
    img = tio.ScalarImage(tensor=output_)
    img.save(test_path)
    print(f"Saved to: {test_path}")

    # Read back and verify
    print("\nReading back saved file...")
    loaded = tio.ScalarImage(test_path)
    loaded_data = loaded.data.numpy()
    print(f"Loaded shape: {loaded_data.shape}")
    print(f"Loaded min: {loaded_data.min():.6f}, max: {loaded_data.max():.6f}, mean: {loaded_data.mean():.6f}")

    break  # Only process first sample
