import os
import time
import torch
import torchvision
import numpy as np
import pytorch_lightning as pl

from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

###

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()

        self.now = now;        self.resume = resume
        self.logdir = logdir;  self.ckptdir = ckptdir;  self.cfgdir = cfgdir
        self.config = config;  self.lightning_config = lightning_config


    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:  # Create logdirs and save configs
            
            for directory in [self.logdir, self.ckptdir, self.cfgdir]:
                os.makedirs(directory, exist_ok=True)

            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:  # ModelCheckpoint callback created log directory --- remove it
            
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                os.makedirs(os.path.split(os.path.join(dst, "child_runs", name))[0], exist_ok=True)

###

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()

        self.max_images = max_images
        self.batch_freq = batch_frequency
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        self.logger_log_images = {pl.loggers.WandbLogger: self._wandb,}

        if not increase_log_steps:  self.log_steps = [self.batch_freq]
        self.clamp = clamp


    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:  # (-1, 1 -> 0, 1) :: [c, h, w]
            grid = (torchvision.utils.make_grid(images[k]) + 1.0) / 2.0 
            pl_module.logger.experiment.add_image(f"{split}/{k}", grid, global_step=pl_module.global_step)


    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)

        for k in images:  # (-1, 1 -> 0, 1) :: [c, h, w]
            grid = (torchvision.utils.make_grid(images[k], nrow=4) + 1.0) / 2.0  
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1).numpy()
            grid = (grid * 255).astype(np.uint8)

            path = os.path.join(root, "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx))
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)


    def log_img(self, pl_module, batch, batch_idx, split="train"):

        if (self.check_frequency(batch_idx) and hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0):

            if pl_module.training:  pl_module.eval()

            with torch.no_grad():
                image_array = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in image_array:
                image_array[k] = image_array[k][:min(image_array[k].shape[0], self.max_images)]

                if isinstance(image_array[k], torch.Tensor):
                    image_array[k] = image_array[k].detach().cpu()
                    if self.clamp:
                        image_array[k] = torch.clamp(image_array[k], -1., 1.)

            logger = type(pl_module.logger)
            self.log_local(pl_module.logger.save_dir, split, image_array, pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, image_array, pl_module.global_step, split)

            if pl_module.training:  pl_module.train()


    def check_frequency(self, batch_idx):
        return (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")

###

class CUDACallback(Callback):  # https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    
    def on_train_epoch_start(self, trainer, pl_module):  # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()


    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")

        except AttributeError:  pass

