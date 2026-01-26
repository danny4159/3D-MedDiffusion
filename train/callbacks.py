

import os
import numpy as np
from PIL import Image

import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torchio as tio





class VolumeLogger(Callback):
    def __init__(self, batch_frequency, max_volumes, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_volumes = max_volumes
        self._tb_logged_epoch = {}
        self.log_steps = [
            2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        # Store fixed samples per modality (CT/MR)
        self.fixed_val_samples = {}  # {"CT": batch, "MR": batch}

    @rank_zero_only
    def log_local(self, save_dir, split, volumes,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "volumes", split)

        
        for k in volumes:
            volumes[k] = (volumes[k] + 1.0)/2.0
            for idx,volume in enumerate(volumes[k]):
                volume = volume.transpose(1,3).transpose(1,2)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}-{}.nii.gz".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx,
                    idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                tio.ScalarImage(tensor = volume).save(path)
    
    def _should_log_tensorboard(self, split: str, epoch: int, modality: str = None) -> bool:
        key = f"{split}_{modality}" if modality else split
        last_epoch = self._tb_logged_epoch.get(key, -1)
        if last_epoch == epoch:
            return False
        self._tb_logged_epoch[key] = epoch
        return True

    def _get_modality(self, path: str) -> str:
        """Infer modality from file path (CT or MR)."""
        path_lower = path.lower()
        if 'ct' in path_lower or 'abdomen' in path_lower:
            return 'CT'
        elif 'mr' in path_lower or 'mri' in path_lower or 't1' in path_lower or 't2' in path_lower or 'brats' in path_lower:
            return 'MR'
        else:
            # Default to CT if unclear
            return 'CT'

    def _mid_slices(self, tensor: torch.Tensor):
        # Expecting tensor shape BxCxDxHxW
        tensor = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
        depth_idx = tensor.shape[2] // 2
        return tensor[:, :, depth_idx, ...]  # BxCxHxW

    @rank_zero_only
    def log_tensorboard(self, pl_module, volumes, split: str, modality: str = None):
        if split != "val":  # Only log validation results
            return
        if not hasattr(pl_module, "logger") or pl_module.logger is None:
            return
        tb_logger = getattr(pl_module.logger, "experiment", None)
        if tb_logger is None or not hasattr(tb_logger, "add_image"):
            return
        if not self._should_log_tensorboard(split, pl_module.current_epoch, modality):
            return

        inputs = volumes.get("inputs")
        outputs = volumes.get("reconstructions")
        targets = volumes.get("gt", volumes.get("target", inputs))
        if inputs is None or outputs is None or targets is None:
            return

        n = min(inputs.shape[0], self.max_volumes)
        input_slices = self._mid_slices(inputs[:n])    # BxCxHxW
        output_slices = self._mid_slices(outputs[:n])  # BxCxHxW
        target_slices = self._mid_slices(targets[:n])  # BxCxHxW

        # Create grid: each row = [input, output, target] for one sample
        # Total rows = n samples, each with 3 images side by side
        rows = []
        for idx in range(n):
            rows.extend([
                input_slices[idx],   # Input (GT)
                output_slices[idx],  # Reconstruction
                target_slices[idx]   # Target (same as input)
            ])

        # Make grid: nrow=3 means 3 columns (input, output, target)
        # Each sample occupies one row
        grid = torchvision.utils.make_grid(rows, nrow=3, padding=4, pad_value=1.0)

        # Log to TensorBoard with modality tag
        if modality:
            tag = f"{split}/{modality}_reconstruction"
        else:
            tag = f"{split}/reconstruction_comparison"
        tb_logger.add_image(tag, grid, global_step=pl_module.global_step)
         
    def log_vid(self, pl_module, batch, batch_idx, split="train"):
        
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_volumes") and
                callable(pl_module.log_volumes) and
                self.max_volumes > 0):
            # print(batch_idx, self.batch_freq,  self.check_frequency(batch_idx))
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                volumes = pl_module.log_volumes(
                    batch, split=split, batch_idx=batch_idx)

            for k in volumes:
                N = min(volumes[k].shape[0], self.max_volumes)
                volumes[k] = volumes[k][:N]
                if isinstance(volumes[k], torch.Tensor):
                    volumes[k] = volumes[k].detach().cpu()
            self.log_tensorboard(pl_module, volumes, split)

            self.log_local(pl_module.logger.save_dir, split, volumes,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx ):
        self.log_vid(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Store fixed validation samples per modality (CT and MR)
        if 'path' in batch:
            path = batch['path'][0] if isinstance(batch['path'], (list, tuple)) else str(batch['path'])
            modality = self._get_modality(path)

            # Store first sample of each modality
            if modality not in self.fixed_val_samples:
                self.fixed_val_samples[modality] = batch
                print(f"[VolumeLogger] Fixed {modality} validation sample stored: {path}")

        # Log all stored modalities on first validation batch
        if batch_idx == 0 and len(self.fixed_val_samples) > 0:
            for modality, fixed_batch in self.fixed_val_samples.items():
                if (self.check_frequency(batch_idx) and
                        hasattr(pl_module, "log_volumes") and
                        callable(pl_module.log_volumes) and
                        self.max_volumes > 0):

                    is_train = pl_module.training
                    if is_train:
                        pl_module.eval()

                    with torch.no_grad():
                        volumes = pl_module.log_volumes(
                            fixed_batch, split="val", batch_idx=batch_idx)

                    for k in volumes:
                        N = min(volumes[k].shape[0], self.max_volumes)
                        volumes[k] = volumes[k][:N]
                        if isinstance(volumes[k], torch.Tensor):
                            volumes[k] = volumes[k].detach().cpu()

                    # Log with modality tag
                    self.log_tensorboard(pl_module, volumes, split="val", modality=modality)
                    self.log_local(pl_module.logger.save_dir, f"val_{modality}", volumes,
                                   pl_module.global_step, pl_module.current_epoch, batch_idx)

                    if is_train:
                        pl_module.train()
