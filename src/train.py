import hydra
import shutil
import os

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

from pathlib import Path

from omegaconf import OmegaConf
import torch

from model import MAEModule

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)

    Path(cfg.checkpoints.dirpath).mkdir(parents=True, exist_ok=True)

    print("Working directory : {}".format(os.getcwd()))

    # copy full config and overrides to checkpoint dir
    shutil.copyfile(
        Path(".hydra/config.yaml"),
        f"{cfg.checkpoint_dir}/config.yaml",
    )
    shutil.copyfile(
        Path(".hydra/overrides.yaml"),
        f"{cfg.checkpoint_dir}/overrides.yaml",
    )

    log_dict = {}

    log_dict["model"] = dict_config["model"]

    log_dict["data"] = dict_config["data"]

    log_dict["trainer"] = dict_config["trainer"]

    seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)

    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoints)

    progress_bar = hydra.utils.instantiate(cfg.progress_bar)

    lr_monitor = LearningRateMonitor()

    callbacks = [
        checkpoint_callback,
        progress_bar,
        lr_monitor,
    ]

    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams(dict_config)
    # Instantiate model and trainer
    model = hydra.utils.instantiate(cfg.model.instance)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    # Resume experiments if last.ckpt exists for this experiment
    ckpt_path = None

    if (Path(cfg.checkpoints.dirpath) / Path("last.ckpt")).exists():
        ckpt_path = Path(cfg.checkpoints.dirpath) / Path("last.ckpt")
    else:
        ckpt_path = None
        if cfg.model.lsuv_normalize:
            from lsuv import lsuv_with_dataloader
            datamodule.setup()
            if isinstance(model, MAEModule):
                #Encoder
                model.encoder = lsuv_with_dataloader(
                    model.encoder,
                    datamodule.train_dataloader(),
                    device=torch.device("cuda:0"),
                    verbose=False,
                )
                torch.nn.init.zeros_(model.encoder.out_proj.weight)
                torch.nn.init.constant_(model.encoder.out_proj.bias, -6.9)
                #Decoder
                model.decoder = lsuv_with_dataloader(
                    model.decoder,
                    datamodule.train_dataloader(),
                    device=torch.device("cuda:0"),
                    verbose=False,
                )
                torch.nn.init.zeros_(model.decoder.out_proj.weight)
                torch.nn.init.constant_(model.decoder.out_proj.bias, -6.9)
            else:
                model.model = lsuv_with_dataloader(
                    model.model,
                    datamodule.train_dataloader(),
                    device=torch.device("cuda:0"),
                    verbose=False,
                )
                torch.nn.init.zeros_(model.out_proj.weight)
                torch.nn.init.constant_(model.out_proj.bias, -6.9)
    # Log activation and gradients if wandb
    if cfg.logger._target_ == "pytorch_lightning.loggers.wandb.WandbLogger":
        logger.experiment.watch(model, log="all", log_graph=True, log_freq=100)

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    train()
