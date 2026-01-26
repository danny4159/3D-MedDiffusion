
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import torchio as tio


class MR2CTDataset(Dataset):
    """
    Dataset for MR to CT translation using ControlNet.

    Args:
        mr_latent_dir: Directory containing MR latent representations
        ct_latent_dir: Directory containing CT latent representations (ground truth)
        split: 'train' or 'val' or 'test'
        resolution: Target latent resolution [D, H, W]
        filter_by_resolution: Whether to filter latents by resolution

    Note:
        - MR and CT latents must have matching filenames (e.g., patient001.nii.gz)
        - Latents should be pre-encoded using generate_training_latent.py
    """
    def __init__(self, mr_latent_dir=None, ct_latent_dir=None, split='train',
                 resolution=[16,64,64], filter_by_resolution=False):
        self.mr_latent_dir = mr_latent_dir
        self.ct_latent_dir = ct_latent_dir
        self.split = split
        self.resolution = resolution
        self.filter_by_resolution = filter_by_resolution

        # Get MR latent file paths
        self.mr_names = glob.glob(os.path.join(mr_latent_dir, './*.nii.gz'), recursive=True)
        self.mr_names.sort()

        # Build corresponding CT paths by matching filenames
        self.ct_names = []
        for mr_path in self.mr_names:
            filename = os.path.basename(mr_path)
            ct_path = os.path.join(ct_latent_dir, filename)
            if os.path.exists(ct_path):
                self.ct_names.append(ct_path)
            else:
                print(f"Warning: No matching CT latent for {filename}")

        # Filter to only keep paired data
        paired_mr = []
        paired_ct = []
        for mr_path in self.mr_names:
            filename = os.path.basename(mr_path)
            ct_path = os.path.join(ct_latent_dir, filename)
            if os.path.exists(ct_path):
                paired_mr.append(mr_path)
                paired_ct.append(ct_path)

        self.mr_names = paired_mr
        self.ct_names = paired_ct

        print(f"Found {len(self.mr_names)} paired MR-CT latent samples")

        # Filter by resolution if requested
        if self.filter_by_resolution:
            filtered_mr = []
            filtered_ct = []
            target_shape = (8, self.resolution[0], self.resolution[1], self.resolution[2])
            for mr_path, ct_path in zip(self.mr_names, self.ct_names):
                img = tio.ScalarImage(mr_path)
                if img.shape == target_shape:
                    filtered_mr.append(mr_path)
                    filtered_ct.append(ct_path)
            print(f"Filtered {len(self.mr_names)} → {len(filtered_mr)} files matching shape {target_shape}")
            self.mr_names = filtered_mr
            self.ct_names = filtered_ct

        # Split into train/val
        self.split = split
        if split == 'train':
            self.mr_names = self.mr_names[:-40]
            self.ct_names = self.ct_names[:-40]
        elif split == 'val':
            self.mr_names = self.mr_names[-40:]
            self.ct_names = self.ct_names[-40:]
        else:  # test
            self.mr_names = self.mr_names[:]
            self.ct_names = self.ct_names[:]

        print(f"[{split}] MR-CT pairs: {len(self.mr_names)}")

    def __len__(self):
        return len(self.mr_names)

    def __getitem__(self, index):
        mr_path = self.mr_names[index]
        ct_path = self.ct_names[index]

        # Load MR latent (condition)
        mr_img = tio.ScalarImage(mr_path)
        mr_data = mr_img.data.to(torch.float32)

        # Load CT latent (ground truth)
        ct_img = tio.ScalarImage(ct_path)
        ct_data = ct_img.data.to(torch.float32)

        if self.split == 'val' or self.split == 'test':
            affine = mr_img.affine  # Use MR affine for reconstruction
            return {
                'data': mr_data,      # MR latent (condition/hint)
                'gt': ct_data,        # CT latent (ground truth)
                'affine': affine,
                'path': mr_path,
                'y': 0                # Class label 0 (single-class CT model)
            }
        else:  # train
            return {
                'data': mr_data,      # MR latent (condition/hint)
                'gt': ct_data,        # CT latent (ground truth)
                'y': 0                # Class label 0 (single-class CT model)
            }
