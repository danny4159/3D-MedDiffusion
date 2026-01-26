# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D MedDiffusion is a medical image generation framework using latent diffusion models for high-quality 3D medical images (CT/MRI). The project implements a novel Patch-Volume Autoencoder with patch-wise encoding and volume-wise decoding, combined with BiFlowNet for diffusion-based generation.

## Environment Setup

```bash
# Create conda environment
conda create -n 3DMedDiffusion python=3.11.11
conda activate 3DMedDiffusion

# Install dependencies
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.2.0 with CUDA 12.1
- PyTorch Lightning 2.5.0
- MONAI 1.4.0 (medical imaging toolkit)
- TorchIO 0.20.3 (medical image preprocessing)

## Architecture

### Two-Stage Training Pipeline

**Stage 1: PatchVolume Autoencoder**
- Compresses 3D medical images into latent space
- Patch-wise encoding, volume-wise decoding
- Supports 4x and 8x compression ratios
- Uses perceptual loss (MedicalNet or LPIPS), GAN loss, and L1 loss
- Training starts discriminator after 20k iterations (`discriminator_iter_start`)

**Stage 2: PatchVolume Autoencoder Refinement**
- Fine-tunes Stage 1 checkpoint
- Requires `resume_from_checkpoint` in config pointing to Stage 1 model

**Latent Encoding**
- Encode training images to latent representations for diffusion training
- Creates latent dataset for BiFlowNet training

**BiFlowNet Diffusion Model**
- Trains on encoded latents
- Uses distributed training with torchrun (multi-GPU)
- Class-conditional generation with DiT-style architecture
- 3D positional embeddings and transformer blocks

### Key Components

**AutoEncoder/** - Patch-Volume Autoencoder implementation
- `model/PatchVolume.py` - Main autoencoder (patchvolumeAE class), discriminator, perceptual loss
- `model/MedicalNetPerceptual.py` - 3D perceptual loss using MedicalNet
- `model/lpips.py` - 2D LPIPS perceptual loss
- `model/codebook.py` - VQ-VAE codebook (if using vector quantization)

**ddpm/** - Diffusion model implementation
- `BiFlowNet.py` - BiFlowNet architecture with GaussianDiffusion class, PatchEmbed_Voxel, transformer blocks

**dataset/** - Data loading
- `vqgan.py`, `vqgan_4x.py` - Dataset classes for 8x and 4x compression
- `Singleres_dataset.py` - Single resolution dataset for BiFlowNet training
- `tr_generate.py` - Dataset for latent generation

**train/** - Training scripts
- `train_PatchVolume.py` - Stage 1 autoencoder training
- `train_PatchVolume_stage2.py` - Stage 2 autoencoder refinement
- `generate_training_latent.py` - Encode images to latents
- `train_BiFlowNet_SingleRes.py` - BiFlowNet diffusion model training
- `callbacks.py` - PyTorch Lightning callbacks (VolumeLogger)

**evaluation/** - Inference scripts
- `class_conditional_generation.py` - Generate with 8x model
- `class_conditional_generation_4x.py` - Generate with 4x model

## Common Commands

### Training

```bash
# Stage 1: Train PatchVolume Autoencoder
# 4x compression
python train/train_PatchVolume.py --config config/PatchVolume_4x.yaml

# 8x compression
python train/train_PatchVolume.py --config config/PatchVolume_8x.yaml

# Stage 2: Refine PatchVolume Autoencoder
# 4x compression
python train/train_PatchVolume_stage2.py --config config/PatchVolume_4x_s2.yaml

# 8x compression
python train/train_PatchVolume_stage2.py --config config/PatchVolume_8x_s2.yaml

# Generate training latents
python train/generate_training_latent.py \
  --data-path config/Singleres_dataset.json \
  --AE-ckpt checkpoints/trained_AE.ckpt \
  --batch-size 4

# Train BiFlowNet (distributed training, 8 GPUs)
torchrun --nnodes=1 --nproc_per_node=8 --master_port 29513 \
  train/train_BiFlowNet_SingleRes.py \
  --data-path config/Singleres_dataset.json \
  --results-dir /path/to/results \
  --num-classes 2 \
  --AE-ckpt /path/to/AE/checkpoint \
  --resolution 32 32 32 \
  --batch-size 48 \
  --num-workers 48
```

### Inference

```bash
# Generation with 8x downsampling model
python evaluation/class_conditional_generation.py \
  --AE-ckpt checkpoints/PatchVolume_8x_s2.ckpt \
  --model-ckpt checkpoints/BiFlowNet_0453500.pt \
  --output-dir /path/to/save/dir

# Generation with 4x downsampling model
python evaluation/class_conditional_generation_4x.py \
  --AE-ckpt checkpoints/PatchVolume_4x_s2.ckpt \
  --model-ckpt checkpoints/BiFlowNet_4x.pt \
  --output-dir /path/to/save/dir
```

## Configuration

Configs use Hydra/OmegaConf structure with base config at `config/base_cfg.yaml`.

### Dataset Configuration

Create JSON files following these formats:

**PatchVolume training** (`config/PatchVolume_data.json`):
```json
{
    "DATA1Name": "Path/to/Data1",
    "DATA2Name": "Path/to/Data2"
}
```

**BiFlowNet training** (`config/Singleres_dataset.json`):
```json
{
    "0": "path/to/class0/data",
    "1": "path/to/class1/data",
    "2": "path/to/class2/data"
}
```

### Key Config Parameters

**Model configs** (`config/PatchVolume_8x.yaml`, etc.):
- `default_root_dir` - Output directory for checkpoints and logs
- `root_dir` - Path to dataset JSON file
- `downsample` - Compression ratio [8,8,8] or [4,4,4]
- `embedding_dim` - Latent embedding dimension (8 for 8x, 16 for 4x typically)
- `n_codes` - Codebook size (8192 default)
- `discriminator_iter_start` - When to start GAN training (20000 default)
- `perceptual_3d` - Use 3D MedicalNet (True) or 2D LPIPS (False)
- `stage` - Training stage (1 or 2)
- `resume_from_checkpoint` - Path to checkpoint (required for Stage 2)

**Dataset configs**:
- `image_channels` - Number of input channels (1 for CT/MRI)
- `imgtype` - Modality type (CT or MRI)
- `patch_size` - Training patch size (64 default)

## Important Implementation Notes

### Data Preprocessing
- All training images must be normalized to [-1, 1] range
- Use `.nii.gz` format for medical images (handled by TorchIO/nibabel)

### Training Stages
1. **Stage 1**: Train autoencoder from scratch with discriminator starting at iteration 20k
2. **Stage 2**: Set `resume_from_checkpoint` to Stage 1 best checkpoint and `stage: 2` in config
3. **Latent Generation**: Use trained autoencoder to encode all training images
4. **BiFlowNet**: Train diffusion model on encoded latents with distributed training

### GPU Requirements
- Autoencoder training: 1+ GPUs with 24GB+ VRAM
- BiFlowNet training: 8 GPUs recommended for distributed training
- Inference: 40GB+ GPU memory for full resolution (512x512x512)

### Distributed Training
- BiFlowNet uses PyTorch DDP via `torchrun`
- Set `--nproc_per_node` to number of GPUs
- Adjust `--batch-size` per-GPU batch size

### Checkpoint Management
- PyTorch Lightning saves checkpoints to `default_root_dir`
- BiFlowNet checkpoints named `BiFlowNet_XXXXXX.pt` (iteration number)
- Always use Stage 2 autoencoder checkpoints for inference

## External Dependencies

**warvito_MedicalNet-models_main/**
- Pretrained MedicalNet models for perceptual loss
- Loaded automatically by `MedicalNetPerceptual` class
- ResNet-based 3D feature extractor trained on medical images

## Project Structure Notes

- Training scripts add parent directory to `sys.path` to import modules
- PyTorch Lightning handles checkpointing, logging, and training loops for autoencoder
- BiFlowNet training uses custom training loop with manual DDP setup
- All image loading uses TorchIO for medical image handling
