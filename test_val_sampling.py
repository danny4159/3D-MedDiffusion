import sys
import os
sys.path.append(os.path.dirname(__file__))

from dataset.vqgan_4x import VQGANDataset_4x
import torch

# Create validation dataset
dataset = VQGANDataset_4x(
    root_dir='config/PatchVolume_data.json',
    augmentation=False,
    split='val',
    stage=1,
    patch_size=64
)

print(f"Total validation samples: {len(dataset)}")
print(f"Target validation patch size: 128×128×64")
print(f"\nTesting validation sampling...")

# Test multiple samples from the same image
patches = []
for i in range(5):
    sample = dataset[0]  # Always index 0
    data = sample['data']
    path = sample['path']
    print(f"\nSample {i+1}:")
    print(f"  Path: {os.path.basename(path)}")
    print(f"  Shape: {data.shape}")
    print(f"  Value range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Mean: {data.mean():.3f}")
    patches.append(data.clone())

# Check if patches are different
print("\n" + "="*60)
print("Checking if validation patches are randomly sampled...")
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
    print("\n✓ Confirmed: Validation patches are randomly sampled")

# Check expected shape
expected_shape = torch.Size([1, 128, 128, 64])
actual_shape = patches[0].shape
if actual_shape == expected_shape:
    print(f"✓ Shape is correct: {actual_shape}")
else:
    print(f"⚠️  Shape mismatch: expected {expected_shape}, got {actual_shape}")
