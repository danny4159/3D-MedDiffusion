
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from AutoEncoder.model.PatchVolume import patchvolumeAE
from train.callbacks import VolumeLogger
from dataset.vqgan_4x import VQGANDataset_4x
from dataset.vqgan import VQGANDataset
import argparse
from omegaconf import OmegaConf

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    pl.seed_everything(cfg.model.seed)
    downsample_ratio = cfg.model.downsample[0]
    if downsample_ratio == 4:
        train_dataset = VQGANDataset_4x(
            root_dir=cfg.dataset.root_dir,augmentation=True,split='train',stage=cfg.model.stage)
        val_dataset = VQGANDataset_4x(
            root_dir=cfg.dataset.root_dir,augmentation=False,split='val')
    else:
        train_dataset = VQGANDataset(
            root_dir=cfg.dataset.root_dir,augmentation=True,split='train',stage=cfg.model.stage)
        val_dataset = VQGANDataset(
            root_dir=cfg.dataset.root_dir,augmentation=False,split='val')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,shuffle=True,
                                  num_workers=cfg.model.num_workers)


    val_dataloader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, num_workers=cfg.model.num_workers)

    # automatically adjust learning rate
    bs, lr, ngpu = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus


    print("Setting learning rate to {:.2e}, batch size to {}, ngpu to {}".format(lr, bs, ngpu))

    model = patchvolumeAE(cfg)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(VolumeLogger(
        batch_frequency=1500, max_volumes=4, clamp=True))



    logger = TensorBoardLogger(cfg.model.default_root_dir, name="my_model")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.model.gpus,
        default_root_dir=cfg.model.default_root_dir,
        strategy='ddp_find_unused_parameters_true',
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        check_val_every_n_epoch=2,
        num_sanity_val_steps = 2,
        log_every_n_steps=10,  # Log more frequently (batches per epoch < 50)
        logger=logger
    )

    if cfg.model.resume_from_checkpoint and os.path.exists(cfg.model.resume_from_checkpoint):
        print('will start from the recent ckpt %s' % cfg.model.resume_from_checkpoint)
        trainer.fit(model, train_dataloader, val_dataloader,ckpt_path=cfg.model.resume_from_checkpoint)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpus", type=str, default=None, help="GPU devices to use, e.g., [1] or [0,1]")
    args, unknown = parser.parse_known_args()

    # Load config and override with CLI args
    cfg = OmegaConf.load(args.config)
    if args.gpus:
        # Parse string like "[1]" or "[0,1]" to list
        import ast
        cfg.model.gpus = ast.literal_eval(args.gpus)

    # Merge with any additional OmegaConf CLI overrides
    cli_conf = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg, cli_conf)

    # Save modified config to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        OmegaConf.save(cfg, f)
        temp_cfg_path = f.name

    main(temp_cfg_path)



