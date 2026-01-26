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
from dataset.Controlnet import ConditionalDataset
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
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
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

def _ddp_dict(_dict):
    new_dict = {}
    for k in _dict:
        new_dict['module.' + k] = _dict[k]
    return new_dict


#################################################################################
#                                  Training Loop                                #
#################################################################################
def get_optimizer_size_in_bytes(optimizer):
    """Calculates the size of the optimizer state in bytes."""
    size_in_bytes = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                size_in_bytes += v.numel() * v.element_size()
    return size_in_bytes

def format_size(bytes_size):
    """Formats the size in bytes into a human-readable string."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024 ** 3:
        return f"{bytes_size / 1024 ** 2:.2f} MB"
    else:
        return f"{bytes_size / 1024 ** 3:.2f} GB"
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def main(args):

    # os.environ['NCCL_SOCKET_IFNAME'] = 'docker0'
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
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        if args.controlnet_ckpt == None:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        else:
            experiment_dir = os.path.dirname(os.path.dirname(args.controlnet_ckpt))
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
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
            # hint_channel=1,
            init_kernel_size=3,
            cond_classes=args.num_classes,
            learn_sigma=False,
            use_sparse_linear_attn=args.use_attn,
            vq_size=args.vq_size,
            num_mid_DiT = args.num_dit,
            patch_size = args.patch_size,
            control_scales = 2.0
        ).cuda()
    
    diffusion= GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).cuda()
    # Note that parameter initialization is done within the DiT constructor
    model.load_pretrained_ldm(args.ldm_ckpt)
    if args.controlnet_ckpt:
        model.load_controlnet_from_ckpt(args.controlnet_ckpt)
        logger.info(f'Using controlnet checkpoint: {args.controlnet_ckpt}')
    else:
        init_with_new_zero, init_with_scratch = model.load_controlnet_from_noise_estimator()
        if rank == 0:
            print(f"strictly load controlnet weight from pretrained SD\n"
                  f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                  f"weights initialized from scratch: {init_with_scratch}")
    model = DDP(model.to(device), device_ids=[rank])
    amp = args.enable_amp
    scaler = GradScaler(enabled=amp)
    if args.AE_ckpt:
        AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt, map_location=f'cuda:{device}')
        AE.eval()
    else:
        raise NotImplementedError()

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")


    opt = torch.optim.Adam(model.module.controlnet.parameters(), lr=1e-4)



    # Setup data:
    train_dataset = ConditionalDataset(condition_dir=args.condition_path, gt_dir=args.gt_path,ref_dir=args.ref_path,split='train', resolution=args.resolution, filter_by_resolution=args.filter_by_resolution)
    val_dataset = ConditionalDataset(condition_dir=args.condition_path, gt_dir=args.gt_path,ref_dir=args.ref_path,split='val', resolution=args.resolution, filter_by_resolution=args.filter_by_resolution)
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
        shuffle=True,
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
    fixed_val_batch = None  # Fixed validation sample for consistent TensorBoard visualization
    logger.info(f"Train Dataset contains {len(train_dataset):,}  Evaluation Dataset contains {len(val_dataset):,} gt images ({args.gt_path}) , recon images ({args.condition_path})")

    # Prepare models for training:
    if args.controlnet_ckpt:
        train_steps = int(os.path.basename(args.controlnet_ckpt).split('.')[0])
        start_epoch = int(train_steps / (len(train_dataset)/(args.batch_size*dist.get_world_size())))
        logger.info(f'Inital state: step = {train_steps}, epoch = {start_epoch}')

    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch,args.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in train_loader:
            x,y,hint = batch['gt'].to(device),batch['y'].to(device),batch['data'].to(device)
            with autocast(enabled=amp):
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                res = torch.tensor([1,1,1]).to(device)/128.0
                res = res.repeat(args.batch_size // dist.get_world_size(), 1)
                loss = diffusion.p_losses(model, x, t, y=y, hint = hint,res=res)
                scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            opt.zero_grad()          

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            # torch.cuda.empty_cache()


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

            # Save Diffusion checkpoint:
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
                    if len(os.listdir(checkpoint_dir))>6:
                        old_ckpt = f"{checkpoint_dir}/{train_steps-3000:07d}.pt"
                        if os.path.exists(old_ckpt):
                            os.remove(old_ckpt)
                    with torch.no_grad():
                        # Use fixed validation sample for consistent visualization
                        if fixed_val_batch is None:
                            fixed_val_batch = next(val_loader_iter)
                            logger.info(f"Fixed validation sample: {fixed_val_batch['path'][0]}")
                        batch = fixed_val_batch
                        x, hint, affine,path,y =  batch['gt'].to(device), batch['data'].to(device), batch['affine'], batch['path'], batch['y'].to(device)
                        milestone = train_steps // args.ckpt_every

                        # Generate sample
                        z = torch.randn_like(x)
                        res = torch.tensor([1,1,1],device=device)/128.0
                        samples = diffusion.p_sample_loop(
                            model, z, y = y,hint=hint,res=res
                        )
                        samples = (((samples + 1.0) / 2.0) * (AE.codebook.embeddings.max() -
                                                            AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()

                        # Decode latents to volumes using sliding window (memory efficient)
                        # For 4x AE: compress_ratio=4, For 8x AE: compress_ratio=8
                        volume_output = AE.decode_sliding(samples,quantize=True,patch_size = 64,sliding_window = 4,compress_ratio = args.compress_ratio)
                        torch.cuda.empty_cache()
                        volume_gt = AE.decode_sliding(x,quantize=True,patch_size = 64,sliding_window = 4,compress_ratio = args.compress_ratio)
                        torch.cuda.empty_cache()
                        volume_condition = AE.decode_sliding(hint,quantize=True,patch_size = 64,sliding_window = 4,compress_ratio = args.compress_ratio)
                        torch.cuda.empty_cache()

                        # Save output volume as .nii.gz
                        name = os.path.basename(path[0]).split('.nii.gz')[0]
                        volume_path = os.path.join(samples_dir,str(f'{milestone}_{name}.nii.gz'))
                        volume_save = volume_output.detach().squeeze(0).cpu()
                        volume_save = volume_save.transpose(1,3).transpose(1,2)
                        tio.ScalarImage(tensor = volume_save,affine = affine[0]).save(volume_path)

                        # TensorBoard: Log central slices (2D)
                        if writer is not None:
                            # Get central slices (axial view, middle slice along z-axis)
                            vol_out = volume_output.detach().cpu()[0, 0]  # (D, H, W)
                            vol_gt_cpu = volume_gt.detach().cpu()[0, 0]
                            vol_cond = volume_condition.detach().cpu()[0, 0]

                            # Use 50% depth ratio for all images to ensure same anatomical position
                            depth_ratio = 0.5
                            mid_d = int(vol_out.shape[0] * depth_ratio)

                            # Extract central slices
                            slice_output = vol_out[mid_d, :, :]  # (H, W)
                            slice_gt = vol_gt_cpu[mid_d, :, :]
                            slice_condition = vol_cond[mid_d, :, :]

                            # Normalize to [0, 1] for visualization
                            slice_output_norm = (slice_output - slice_output.min()) / (slice_output.max() - slice_output.min() + 1e-8)
                            slice_gt_norm = (slice_gt - slice_gt.min()) / (slice_gt.max() - slice_gt.min() + 1e-8)
                            slice_condition_norm = (slice_condition - slice_condition.min()) / (slice_condition.max() - slice_condition.min() + 1e-8)

                            # Log to TensorBoard
                            writer.add_image(f'val/condition', slice_condition_norm.unsqueeze(0), train_steps)
                            writer.add_image(f'val/gt', slice_gt_norm.unsqueeze(0), train_steps)
                            writer.add_image(f'val/output', slice_output_norm.unsqueeze(0), train_steps)

                            # Try to load original (non-latent) images
                            filename = os.path.basename(path[0])
                            condition_orig_path = path[0].replace('_latents', '')
                            gt_orig_path = os.path.join(args.gt_path.replace('_latents', ''), filename)

                            try:
                                if os.path.exists(condition_orig_path):
                                    cond_orig_img = tio.ScalarImage(condition_orig_path)
                                    vol_cond_orig = cond_orig_img.data.squeeze(0)  # (D, H, W)
                                    # Match decode_sliding output shape by permuting: (D, H, W) → (W, D, H)
                                    vol_cond_orig = vol_cond_orig.permute(2, 0, 1)
                                    mid_d_cond = int(vol_cond_orig.shape[0] * depth_ratio)
                                    slice_cond_orig = vol_cond_orig[mid_d_cond, :, :]
                                    slice_cond_orig_norm = (slice_cond_orig - slice_cond_orig.min()) / (slice_cond_orig.max() - slice_cond_orig.min() + 1e-8)
                                    writer.add_image(f'val/condition_original', slice_cond_orig_norm.unsqueeze(0), train_steps)
                            except Exception as e:
                                pass

                            try:
                                if os.path.exists(gt_orig_path):
                                    gt_orig_img = tio.ScalarImage(gt_orig_path)
                                    vol_gt_orig = gt_orig_img.data.squeeze(0)  # (D, H, W)
                                    # Match decode_sliding output shape by permuting: (D, H, W) → (W, D, H)
                                    vol_gt_orig = vol_gt_orig.permute(2, 0, 1)
                                    mid_d_gt = int(vol_gt_orig.shape[0] * depth_ratio)
                                    slice_gt_orig = vol_gt_orig[mid_d_gt, :, :]
                                    slice_gt_orig_norm = (slice_gt_orig - slice_gt_orig.min()) / (slice_gt_orig.max() - slice_gt_orig.min() + 1e-8)
                                    writer.add_image(f'val/gt_original', slice_gt_orig_norm.unsqueeze(0), train_steps)
                            except Exception as e:
                                pass

                            # Create comparison grid (condition_latent | gt_latent | output)
                            comparison = torch.cat([slice_condition_norm.unsqueeze(0),
                                                   slice_gt_norm.unsqueeze(0),
                                                   slice_output_norm.unsqueeze(0)], dim=2)
                            writer.add_image(f'val/comparison_latent', comparison, train_steps)

                            # Create comparison grid with original images if available
                            try:
                                if os.path.exists(condition_orig_path) and os.path.exists(gt_orig_path):
                                    cond_orig_img = tio.ScalarImage(condition_orig_path)
                                    gt_orig_img = tio.ScalarImage(gt_orig_path)
                                    vol_cond_orig = cond_orig_img.data.squeeze(0)  # (D, H, W)
                                    vol_gt_orig = gt_orig_img.data.squeeze(0)      # (D, H, W)
                                    # Match decode_sliding output shape by permuting: (D, H, W) → (W, D, H)
                                    vol_cond_orig = vol_cond_orig.permute(2, 0, 1)
                                    vol_gt_orig = vol_gt_orig.permute(2, 0, 1)
                                    mid_d_cond_comp = int(vol_cond_orig.shape[0] * depth_ratio)
                                    mid_d_gt_comp = int(vol_gt_orig.shape[0] * depth_ratio)
                                    slice_cond_orig = vol_cond_orig[mid_d_cond_comp, :, :]
                                    slice_gt_orig = vol_gt_orig[mid_d_gt_comp, :, :]
                                    slice_cond_orig_norm = (slice_cond_orig - slice_cond_orig.min()) / (slice_cond_orig.max() - slice_cond_orig.min() + 1e-8)
                                    slice_gt_orig_norm = (slice_gt_orig - slice_gt_orig.min()) / (slice_gt_orig.max() - slice_gt_orig.min() + 1e-8)
                                    comparison_orig = torch.cat([slice_cond_orig_norm.unsqueeze(0),
                                                               slice_gt_orig_norm.unsqueeze(0),
                                                               slice_output_norm.unsqueeze(0)], dim=2)
                                    writer.add_image(f'val/comparison_original', comparison_orig, train_steps)
                            except Exception as e:
                                pass
                dist.barrier()
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":







    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-path", type=str, required=True)
    parser.add_argument("--gt-path", type=str, required=True)
    parser.add_argument("--ref-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="")
    parser.add_argument("--loss-type", type=str, default='l1')
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model-dim", type=int, default=72)
    parser.add_argument("--dim-mults", type=list, default=[1,1,2,4,8])
    parser.add_argument("--use-attn", type=list, default=[0,0,0,1,1])
    parser.add_argument("--enable_amp", type=bool, default=False)
    parser.add_argument("--model", type=str,default="Controlnet")
    parser.add_argument("--AE-ckpt", type=str, default="")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8) # 48 for B/4 ; 24 for L/4 shiyixia 
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8) ##
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--ldm-ckpt", type=str, default='')
    parser.add_argument("--controlnet-ckpt", type=str, default=None)
    parser.add_argument("--vq-size", type=int, default=64)

    parser.add_argument("--num-dit", type=int, default=1, help="Number of middle DiT blocks")
    parser.add_argument("--patch-size", type=int, default=2, help="Patch size for the model")
    parser.add_argument("--resolution", type=int, nargs=3, default=[16, 64, 64], help="Target resolution for latents (z, h, w)")
    parser.add_argument("--compress-ratio", type=int, default=4, help="Autoencoder compression ratio (4 for 4x AE, 8 for 8x AE)")
    parser.add_argument("--filter-by-resolution", action='store_true', help="Only use latents matching the specified resolution")
    args = parser.parse_args()
    main(args)

