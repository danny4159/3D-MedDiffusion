import sys
import os
sys.path.append(os.path.dirname(__file__))

from dataset.vqgan_4x import VQGANDataset_4x
import torch
import numpy as np

# Create dataset
dataset = VQGANDataset_4x(
    root_dir='config/PatchVolume_data.json',
    augmentation=False,
    split='train',
    stage=1,
    patch_size=64
)

print(f"Total training samples: {len(dataset)}")
print(f"Testing first sample with index 0...")

# Get same sample 5 times to check randomness
patches = []
for i in range(5):
    sample = dataset[0]  # Always index 0
    data = sample['data']  # Shape: (C, D, H, W) after transpose
    print(f"\nSample {i+1}:")
    print(f"  Shape: {data.shape}")
    print(f"  Value range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Mean: {data.mean():.3f}")
    print(f"  First 5 values: {data.flatten()[:5]}")
    patches.append(data.clone())

# Check if patches are different (random sampling)
print("\n" + "="*60)
print("Checking if patches are randomly sampled...")
print("="*60)

all_same = True
for i in range(1, 5):
    diff = (patches[0] - patches[i]).abs().sum().item()
    if diff > 1e-6:
        all_same = False
        print(f"Patch 1 vs Patch {i+1}: Different (diff={diff:.2f})")
    else:
        print(f"Patch 1 vs Patch {i+1}: Same")

if all_same:
    print("\n⚠️  WARNING: All patches are identical - NOT random sampling!")
else:
    print("\n✓ Confirmed: Patches are randomly sampled from different locations")
