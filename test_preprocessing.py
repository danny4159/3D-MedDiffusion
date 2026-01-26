import sys
import os
sys.path.append(os.path.dirname(__file__))

from dataset.vqgan_4x import VQGANDataset_4x
import torch

# Create dataset
dataset = VQGANDataset_4x(
    root_dir='config/PatchVolume_data.json',
    augmentation=False,
    split='train',
    stage=1,
    patch_size=64
)

print("="*60)
print("Testing preprocessing order and consistency")
print("="*60)

# Get multiple patches from the same image
patches = []
for i in range(10):
    sample = dataset[0]  # Same image
    data = sample['data']
    patches.append(data.clone())

    if i == 0:
        print(f"\nFirst patch:")
        print(f"  Shape: {data.shape}")
        print(f"  Value range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Mean: {data.mean():.3f}")

# Collect all unique values from all patches to check value distribution
all_values = torch.cat([p.flatten() for p in patches])
print(f"\n10 patches combined statistics:")
print(f"  Total values: {len(all_values)}")
print(f"  Overall range: [{all_values.min():.3f}, {all_values.max():.3f}]")
print(f"  Overall mean: {all_values.mean():.3f}")

# Check background value consistency
background_count = (all_values < -0.99).sum().item()
print(f"\n  Values close to -1.0 (background): {background_count} ({100*background_count/len(all_values):.1f}%)")

# Check if values are in expected range
in_range = ((all_values >= -1.0) & (all_values <= 1.0)).all()
print(f"\n✓ All values in [-1, 1]: {in_range}")

# Show value distribution
print(f"\nValue distribution:")
print(f"  < -0.5: {(all_values < -0.5).sum().item()}")
print(f"  -0.5 to 0: {((all_values >= -0.5) & (all_values < 0)).sum().item()}")
print(f"  0 to 0.5: {((all_values >= 0) & (all_values < 0.5)).sum().item()}")
print(f"  >= 0.5: {(all_values >= 0.5).sum().item()}")

print("\n" + "="*60)
print("Expected behavior:")
print("  - All values should be in [-1, 1]")
print("  - Background (air) should be close to -1.0")
print("  - Tissue values should be consistent across patches")
print("  - Padding areas (if any) should be -1.0")
print("="*60)
