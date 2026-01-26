import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.MR2CT_dataset import MR2CTDataset
from collections import OrderedDict
from glob import glob
from time import time
import argparse
import logging
import os
from ddpm.cldm import ControlLDM
from ddpm.BiFlowNet import GaussianDiffusion
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio
from torch.cuda.amp import autocast, GradScaler
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        if args.controlnet_ckpt == None:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-MR2CT-Controlnet"
        else:
            experiment_dir = os.path.dirname(os.path.dirname(args.controlnet_ckpt))
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        samples_dir = f"{experiment_dir}/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok= True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # Setup TensorBoard writer
        writer = SummaryWriter(log_dir=experiment_dir)
        logger.info(f"TensorBoard logging to {experiment_dir}")
    else:
        logger = create_logger(None)
        writer = None



    model = ControlLDM(
            dim=args.model_dim,
            dim_mults=args.dim_mults,
            channels=args.volume_channels,
            init_kernel_size=3,
            cond_classes=args.num_classes,
            learn_sigma=False,
            use_sparse_linear_attn=args.use_attn,
            vq_size=args.vq_size,
            num_mid_DiT = args.num_dit,
            patch_size = args.patch_size,
            control_scales = args.control_scale
        ).cuda()

    diffusion= GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).cuda()

    # Load pretrained LDM (BiFlowNet)
    model.load_pretrained_ldm(args.ldm_ckpt)
    logger.info(f'Loaded pretrained LDM from: {args.ldm_ckpt}')

    if args.controlnet_ckpt:
        model.load_controlnet_from_ckpt(args.controlnet_ckpt)
        logger.info(f'Using controlnet checkpoint: {args.controlnet_ckpt}')
    else:
        init_with_new_zero, init_with_scratch = model.load_controlnet_from_noise_estimator()
        if rank == 0:
            print(f"Initialized ControlNet from pretrained LDM\n"
                  f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                  f"weights initialized from scratch: {init_with_scratch}")

    model = DDP(model.to(device), device_ids=[rank])
    amp = args.enable_amp
    scaler = GradScaler(enabled=amp)

    # Load AutoEncoder for validation decoding
    if args.AE_ckpt:
        AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt, map_location=f'cuda:{device}')
        AE.eval()
    else:
        raise ValueError("AE checkpoint required for validation!")

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.module.controlnet.parameters(), lr=args.lr)

    # Setup data:
    train_dataset = MR2CTDataset(
        mr_latent_dir=args.mr_latent_path,
        ct_latent_dir=args.ct_latent_path,
        split='train',
        resolution=args.resolution,
        filter_by_resolution=args.filter_by_resolution
    )
    val_dataset = MR2CTDataset(
        mr_latent_dir=args.mr_latent_path,
        ct_latent_dir=args.ct_latent_path,
        split='val',
        resolution=args.resolution,
        filter_by_resolution=args.filter_by_resolution
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,  # Don't shuffle for consistent validation
        seed=args.global_seed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader_iter = cycle(val_loader)
    fixed_val_batch = None  # Fixed validation sample
    logger.info(f"Train Dataset contains {len(train_dataset):,}, Val Dataset contains {len(val_dataset):,} MR-CT pairs")

    # Prepare models for training:
    if args.controlnet_ckpt:
        train_steps = int(os.path.basename(args.controlnet_ckpt).split('.')[0])
        start_epoch = int(train_steps / (len(train_dataset)/(args.batch_size*dist.get_world_size())))
        logger.info(f'Initial state: step = {train_steps}, epoch = {start_epoch}')

    model.train()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in train_loader:
            mr_latent = batch['data'].to(device)     # MR latent (condition)
            ct_latent = batch['gt'].to(device)       # CT latent (ground truth)
            y = batch['y'].to(device)                # Class label (1 for CT)

            with autocast(enabled=amp):
                # Random timestep
                t = torch.randint(0, diffusion.num_timesteps, (ct_latent.shape[0],), device=device)

                # Resolution conditioning (optional, can be fixed)
                res = torch.tensor([1,1,1]).to(device)/128.0
                res = res.repeat(args.batch_size // dist.get_world_size(), 1)

                # ControlNet training: condition on MR, predict noise in CT
                loss = diffusion.p_losses(
                    model,
                    x=ct_latent,      # Target: CT latent
                    t=t,
                    y=y,              # Class: CT (1)
                    hint=mr_latent,   # Condition: MR latent
                    res=res
                )
                scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # TensorBoard logging
                if rank == 0 and writer is not None:
                    writer.add_scalar('train/loss', avg_loss, train_steps)
                    writer.add_scalar('train/steps_per_sec', steps_per_sec, train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint and run validation:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.controlnet.state_dict(),
                        "scaler": scaler.state_dict(),
                        "opt":opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    # Remove old checkpoints (keep last 6)
                    if len(os.listdir(checkpoint_dir)) > 6:
                        old_ckpt = f"{checkpoint_dir}/{train_steps-3000:07d}.pt"
                        if os.path.exists(old_ckpt):
                            os.remove(old_ckpt)

                    # Validation: Generate CT from MR
                    with torch.no_grad():
                        # Use fixed validation sample
                        if fixed_val_batch is None:
                            fixed_val_batch = next(val_loader_iter)
                            logger.info(f"Fixed validation sample: {fixed_val_batch['path'][0]}")

                        batch = fixed_val_batch
                        mr_latent = batch['data'].to(device)
                        ct_latent_gt = batch['gt'].to(device)
                        affine = batch['affine']
                        path = batch['path']
                        y = batch['y'].to(device)

                        milestone = train_steps // args.ckpt_every

                        # Generate CT latent from MR (with fixed seed for consistency)
                        torch.manual_seed(args.global_seed)
                        z = torch.randn_like(ct_latent_gt)
                        res = torch.tensor([1,1,1], device=device)/128.0

                        # Sample using ControlNet (MR as condition)
                        ct_latent_pred = diffusion.p_sample_loop(
                            model,
                            z,
                            y=y,
                            hint=mr_latent,  # MR latent as condition
                            res=res
                        )

                        # Denormalize latents
                        ct_latent_pred = (((ct_latent_pred + 1.0) / 2.0) *
                                         (AE.codebook.embeddings.max() - AE.codebook.embeddings.min())) + \
                                         AE.codebook.embeddings.min()
                        ct_latent_gt = (((ct_latent_gt + 1.0) / 2.0) *
                                       (AE.codebook.embeddings.max() - AE.codebook.embeddings.min())) + \
                                       AE.codebook.embeddings.min()
                        mr_latent = (((mr_latent + 1.0) / 2.0) *
                                    (AE.codebook.embeddings.max() - AE.codebook.embeddings.min())) + \
                                    AE.codebook.embeddings.min()

                        # Decode latents to volumes (patch_size=64 for consistency with BiFlowNet)
                        ct_pred_volume = AE.decode_sliding(ct_latent_pred, quantize=True,
                                                          patch_size=64, sliding_window=2, compress_ratio=8)
                        torch.cuda.empty_cache()
                        ct_gt_volume = AE.decode_sliding(ct_latent_gt, quantize=True,
                                                        patch_size=64, sliding_window=2, compress_ratio=8)
                        torch.cuda.empty_cache()
                        mr_volume = AE.decode_sliding(mr_latent, quantize=True,
                                                     patch_size=64, sliding_window=2, compress_ratio=8)
                        torch.cuda.empty_cache()

                        # Save predicted CT volume
                        name = os.path.basename(path[0]).split('.nii.gz')[0]
                        volume_path = os.path.join(samples_dir, f'{milestone}_{name}_pred_CT.nii.gz')
                        volume_save = ct_pred_volume.detach().squeeze(0).cpu()
                        volume_save = volume_save.transpose(1,3).transpose(1,2)
                        tio.ScalarImage(tensor=volume_save, affine=affine[0]).save(volume_path)

                        # TensorBoard: Log central slices
                        if writer is not None:
                            # Get central slices (axial view)
                            mr_cpu = mr_volume.detach().cpu()[0, 0]
                            ct_gt_cpu = ct_gt_volume.detach().cpu()[0, 0]
                            ct_pred_cpu = ct_pred_volume.detach().cpu()[0, 0]

                            mid_d = mr_cpu.shape[0] // 2

                            # Extract slices
                            slice_mr = mr_cpu[mid_d, :, :]
                            slice_ct_gt = ct_gt_cpu[mid_d, :, :]
                            slice_ct_pred = ct_pred_cpu[mid_d, :, :]

                            # Normalize to [0, 1]
                            slice_mr_norm = (slice_mr - slice_mr.min()) / (slice_mr.max() - slice_mr.min() + 1e-8)
                            slice_ct_gt_norm = (slice_ct_gt - slice_ct_gt.min()) / (slice_ct_gt.max() - slice_ct_gt.min() + 1e-8)
                            slice_ct_pred_norm = (slice_ct_pred - slice_ct_pred.min()) / (slice_ct_pred.max() - slice_ct_pred.min() + 1e-8)

                            # Log individual images
                            writer.add_image(f'val/MR_input', slice_mr_norm.unsqueeze(0), train_steps)
                            writer.add_image(f'val/CT_ground_truth', slice_ct_gt_norm.unsqueeze(0), train_steps)
                            writer.add_image(f'val/CT_predicted', slice_ct_pred_norm.unsqueeze(0), train_steps)

                            # Create comparison grid (MR | GT CT | Pred CT)
                            comparison = torch.cat([slice_mr_norm.unsqueeze(0),
                                                   slice_ct_gt_norm.unsqueeze(0),
                                                   slice_ct_pred_norm.unsqueeze(0)], dim=2)
                            writer.add_image(f'val/MR2CT_comparison', comparison, train_steps)

                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr-latent-path", type=str, required=True, help="Path to MR latents")
    parser.add_argument("--ct-latent-path", type=str, required=True, help="Path to CT latents (ground truth)")
    parser.add_argument("--results-dir", type=str, default="results/MR2CT_ControlNet")
    parser.add_argument("--loss-type", type=str, default='l1')
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model-dim", type=int, default=72)
    parser.add_argument("--dim-mults", type=list, default=[1,1,2,4,8])
    parser.add_argument("--use-attn", type=list, default=[0,0,0,1,1])
    parser.add_argument("--enable_amp", type=bool, default=False)
    parser.add_argument("--AE-ckpt", type=str, required=True, help="PatchVolume AE checkpoint")
    parser.add_argument("--ldm-ckpt", type=str, required=True, help="Pretrained BiFlowNet checkpoint")
    parser.add_argument("--controlnet-ckpt", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes (1=CT only)")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--vq-size", type=int, default=64)
    parser.add_argument("--num-dit", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, nargs=3, default=[16, 64, 64])
    parser.add_argument("--filter-by-resolution", action='store_true')
    parser.add_argument("--control-scale", type=float, default=1.0, help="ControlNet influence scale")
    args = parser.parse_args()
    main(args)
