
import torch
from torch.utils.data.dataset import Dataset
import os
import random
import glob
import torchio as tio
import json
import random

class VQGANDataset(Dataset):
    def __init__(self, root_dir=None, augmentation=False,split='train',stage = 1,patch_size = 64):
        randnum = 216
        self.file_names = []
        self.stage = stage
        print(root_dir)
        if root_dir.endswith('json'):
            with open(root_dir) as json_file:
                dataroots = json.load(json_file)
            for key,value in dataroots.items():
                if type(value) == list:
                    for path in value:
                        self.file_names += (glob.glob(os.path.join(path, './*.nii.gz'), recursive=True))
                else:
                    self.file_names += (glob.glob(os.path.join(value, './*.nii.gz'), recursive=True))
        else:
            self.root_dir = root_dir
            self.file_names = glob.glob(os.path.join(
                        root_dir, './*.nii.gz'), recursive=True)
        random.seed(randnum)
        random.shuffle(self.file_names )

        self.split = split
        self.augmentation = augmentation
        if split == 'train':
            self.file_names = self.file_names[:-40]
        elif split == 'val':
            self.file_names = self.file_names[-40:]
            self.augmentation = False
        self.patch_sampler = tio.data.UniformSampler(patch_size)
        self.patch_sampler_192 = tio.data.UniformSampler((192,192,64))  # For Stage 2 train
        self.patch_sampler_256 = tio.data.UniformSampler((256,256,64))  # For Stage 1 val (same as train)
        self.randomflip = tio.RandomFlip( axes=(0,1),flip_probability=0.5)
        print(f'With patch size {str(patch_size)}')
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        whole_img = tio.ScalarImage(path)

        # Track original shape
        original_shape = whole_img.shape[1:]  # (H, W, D)

        # Step 1: Normalize to [0, 1] using min-max normalization
        img_min = whole_img.data.min()
        img_max = whole_img.data.max()
        if img_max > img_min:  # Avoid division by zero
            whole_img.data = (whole_img.data - img_min) / (img_max - img_min)
        else:
            whole_img.data = torch.zeros_like(whole_img.data)

        # Step 2: Convert [0, 1] to [-1, 1] BEFORE padding/patching (so padding becomes -1)
        whole_img.data = whole_img.data * 2 - 1

        # Step 3: Apply padding and patching
        if self.stage == 1 and self.split == 'train':
            patch_size = self.patch_sampler.patch_size
            current_shape = whole_img.shape[1:]  # (H, W, D)
            padding = []
            for current, target in zip(current_shape, patch_size):
                if current < target:
                    pad_total = target - current
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    padding.extend([pad_before, pad_after])
                else:
                    padding.extend([0, 0])

            if any(p > 0 for p in padding):
                whole_img = tio.Pad(padding, padding_mode=-1)(whole_img)  # Pad with -1 (background)

            padded_shape = whole_img.shape[1:]

            img = None
            while img== None or img.data.sum() ==0:
                img = next(self.patch_sampler(tio.Subject(image = whole_img)))['image']

            sampled_shape = img.shape[1:]
            # print(f"[Stage1-Train] Original: {original_shape} -> Padded: {padded_shape} -> Sampled: {sampled_shape}")

        elif self.stage ==2 and self.split == 'train':
            img = whole_img
            target_size = self.patch_sampler_192.patch_size
            current_shape = img.shape[1:]
            padding = []
            needs_padding = False
            for current, target in zip(current_shape, target_size):
                if current < target:
                    needs_padding = True
                    pad_total = target - current
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    padding.extend([pad_before, pad_after])
                else:
                    padding.extend([0, 0])

            if needs_padding:
                img = tio.Pad(padding, padding_mode=-1)(img)  # Pad with -1 (background)

            padded_shape = img.shape[1:]

            # Always do patch sampling for stage 2 (needed for unfold operation)
            img = next(self.patch_sampler_192(tio.Subject(image = img)))['image']

            sampled_shape = img.shape[1:]
            # print(f"[Stage2-Train] Original: {original_shape} -> Padded: {padded_shape} -> Sampled: {sampled_shape}")

        elif self.split =='val':
            img = whole_img
            target_size = self.patch_sampler_256.patch_size
            current_shape = img.shape[1:]

            # Pad only dimensions that are smaller than target
            padding = []
            for current, target in zip(current_shape, target_size):
                if current < target:
                    pad_total = target - current
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    padding.extend([pad_before, pad_after])
                else:
                    padding.extend([0, 0])

            # Apply padding if any dimension needs it
            if any(p > 0 for p in padding):
                img = tio.Pad(padding, padding_mode=-1)(img)  # Pad with -1 (background)

            padded_shape = img.shape[1:]

            # Always do random patch sampling (after padding, size is guaranteed >= target)
            img = next(self.patch_sampler_256(tio.Subject(image = img)))['image']

            sampled_shape = img.shape[1:]
            # print(f"[Val] Original: {original_shape} -> Padded: {padded_shape} -> Sampled: {sampled_shape}", flush=True)

        if self.augmentation:
            img = self.randomflip(img)
        imageout = img.data
        if self.augmentation and random.random()>0.5:
            imageout = torch.rot90(imageout,dims=(1,2))

        # Already converted to [-1, 1] before patching, so just transpose
        imageout = imageout.transpose(1,3).transpose(2,3)
        imageout = imageout.type(torch.float32)

        if self.split =='val':
            return {'data': imageout , 'affine' : img.affine , 'path':path}
        else:
            return {'data': imageout}
