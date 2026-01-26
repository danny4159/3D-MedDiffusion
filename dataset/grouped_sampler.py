import torch
from torch.utils.data import Sampler
import torchio as tio
from collections import defaultdict
import random


class GroupedBySizeSampler(Sampler):
    """
    Sampler that groups samples by similar sizes to minimize padding.
    Samples within each batch will have the same or very similar sizes.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by latent size
        self.size_groups = defaultdict(list)
        print("Grouping dataset by latent sizes...")

        for idx in range(len(dataset)):
            # Get the file path from dataset
            file_info = list(dataset.all_files[idx].items())[0]
            file_path = file_info[1]

            # Load to get shape
            latent = tio.ScalarImage(file_path)
            shape = latent.shape[1]  # depth dimension

            self.size_groups[shape].append(idx)

        # Print statistics
        print(f"Found {len(self.size_groups)} different sizes:")
        for size, indices in sorted(self.size_groups.items()):
            print(f"  Size {size}: {len(indices)} samples")

        # Create batches
        self.batches = []
        for size, indices in self.size_groups.items():
            if self.shuffle:
                random.shuffle(indices)

            # Create batches from this size group
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                self.batches.append(batch)

        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)

        for batch in self.batches:
            yield from batch

    def __len__(self):
        return len(self.dataset)
