"""
# Created: 2023-07-12 19:30
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Train Model
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint
)

from omegaconf import DictConfig, OmegaConf
import hydra, os
from pathlib import Path
#! change dataloader
from scripts.network.dataloader_flow4D import HDF5Dataset_multi_frame_idx, collate_fn_pad_less_to_more
from scripts.pl_model_flow4D import ModelWrapper

from datetime import datetime
from torch.utils.data.dataset import random_split
from hydra.core.hydra_config import HydraConfig

@hydra.main(version_base=None, config_path="conf", config_name="config_flow4D")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    # current_time = datetime.now().strftime("%m%d_%H%M%S")
    # cfg.output = current_time
    # output_dir = os.path.join('logs/wandb', current_time)
    output_dir = HydraConfig.get().runtime.output_dir
    print('output_dir = {}'.format(output_dir)) 
    print('val_check = {}'.format(cfg.val_every))
    print('monitor={}'.format(cfg.model.val_monitor))

    
    if cfg.debug == True: 
        train_dataset = HDF5Dataset_multi_frame_idx(cfg.dataset_path + "/train_debug", cfg.num_frames) 
        val_dataset = HDF5Dataset_multi_frame_idx(cfg.dataset_path + "/val", cfg.num_frames)
        cfg.num_workers = 1
        total_size = len(val_dataset)
        subset_size = total_size // 100 
        _, val_dataset = random_split(val_dataset, [total_size - subset_size, subset_size])
        print('debug mode')
    else:
        train_dataset = HDF5Dataset_multi_frame_idx(cfg.dataset_path + "/train", cfg.num_frames)
        val_dataset = HDF5Dataset_multi_frame_idx(cfg.dataset_path + "/val", cfg.num_frames)
        print('full training mode')
    
    # count gpus, overwrite gpus
    cfg.gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=True, # false for debug
                            num_workers=cfg.num_workers,
                            collate_fn=collate_fn_pad_less_to_more,
                            pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            collate_fn=collate_fn_pad_less_to_more,
                            pin_memory=True)

    
    Path(os.path.join(output_dir, "checkpoints")).mkdir(parents=True, exist_ok=True)
    
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    model = ModelWrapper(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch:02d}_" + "_{val/Dynamic/Mean:.3f}_{val/Static/Mean:.3f}_{val/Three-way:.3f}_{val/EPE_FD:.3f}_{val/EPE_BS:.3f}_{val/EPE_FS:.3f}_{val/IoU:.2f}",
            auto_insert_metric_name=False,
            monitor=cfg.model.val_monitor,
            mode="min",
            save_top_k=cfg.save_top_model,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]

    cfg.wandb_mode = "disabled"
    if cfg.wandb_mode != "disabled":
        wandb_logger = WandbLogger(save_dir=output_dir,
                                #    entity="kth-rpl",
                                project=f"{cfg.wandb_project_name}", 
                                name=f"{cfg.output}",
                                offline=(cfg.wandb_mode == "offline"),
                                log_model=(True if cfg.wandb_mode == "online" else False))
        wandb_logger.watch(model, log_graph=False)
    else:
        wandb_logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    trainer = pl.Trainer(log_every_n_steps=50,
                         logger=wandb_logger,
                         accelerator="gpu",
                         devices=cfg.gpus,
                         check_val_every_n_epoch=cfg.val_every,
                         gradient_clip_val=cfg.gradient_clip_val,
                         strategy="ddp_find_unused_parameters_false" if cfg.gpus > 1 else "auto", #!change for debug
                         callbacks=callbacks,
                         max_epochs=cfg.epochs,
                         sync_batchnorm=cfg.sync_bn)

    

    if trainer.global_rank == 0:
        print("\n"+"-"*40)
        print("Initiating wandb and trainer successfully.  ^V^ ")
        print(f"We will use {cfg.gpus} GPUs to train the model. Check the checkpoints in {output_dir} checkpoints folder.")
        print("Total Train Dataset Size: ", len(train_dataset))
        print("-"*40+"\n")

    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader, ckpt_path = cfg.checkpoint)

if __name__ == "__main__":
    main()