import numpy as np
import lightning.pytorch as L
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torchvision.transforms import transforms

from .sampler.sampler import DPMScheduler, VideoFlowMatchingSampler

denormalize = transforms.Normalize(
    mean=[-1],
    std=[2.],
)


class VideoDiffusionModule(L.LightningModule):
    def __init__(
            self,
            model,
            mode,
            loss,
            optimizer_cfg,
            lr_scheduler_builder,
            torch_compile=False,
            ema_cfg=None,
            block_causal=False
        ):
        super().__init__()
        # do optim
        if torch_compile:
            print("compiling model")
            model = torch.compile(model, mode="max-autotune-no-cudagraphs")
        self.model = model
        self.mode = mode
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_builder = lr_scheduler_builder

        self.temporal_mask = None
        if block_causal:
            self.temporal_mask = self.model.make_block_causal_temporal_mask()
            print("using block causal temporal mask")

        #ema
        if ema_cfg is not None:
            from ema_pytorch import EMA
            self.ema_cfg = ema_cfg
            self.ema = EMA(
                model,
                beta = ema_cfg.beta,
                update_after_step= ema_cfg.update_after_step,
                update_every= ema_cfg.update_every,
                include_online_model=False
            )

        # noise scheduler
        self.n_timesteps = model.n_timesteps
        print(f"prediction mode: {self.mode}")
        self.sampler = VideoFlowMatchingSampler(model=self.ema.ema_model) if self.mode == "fm" else DPMScheduler(model=self.ema.ema_model)


    def training_step(self, batch, batch_idx):
        vid, txt, mask = batch
        b, n = mask.shape

        # drop labels
        drop = torch.rand(b, device=txt.device) < 0.1
        drop = drop.unsqueeze(-1)
        zero = torch.zeros_like(mask)
        mask = torch.where(drop, zero, mask)

        #sample time, noise, make noisy
        # each sample gets a noise between i/b and i/(b=1) to have uniform time in batch
        # time = torch.linspace(0, (b-1)/b, b) + torch.rand(b)/b
        # time = (time*self.n_timesteps).to(img.device)
        time = torch.randint(0, self.n_timesteps, (b,)).to(vid.device)
        eps = torch.randn_like(vid)
        n_img = self.sampler.add_noise(vid, eps, time)

        pred = self.model(n_img, time, txt, mask, temporal_mask=self.temporal_mask)

        if self.mode == "fm":
            target = (eps - vid)
            loss = {"loss": ((target - pred)**2).mean()}
        elif self.mode == "eps":
            loss = {"loss": ((pred - eps)**2).mean()}

        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar = True
            )

        #ema
        if self.ema is not None:
            self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):

        vid, txt, mask = batch
        b, n = mask.shape

        # print(f"vid: {vid.shape} txt: {txt.shape} mask: {mask.shape}")
        #sample time, noise, make noisy
        # each sample gets a noise between i/b and i/(b=1) to have uniform time in batch
        time = torch.linspace(0, self.n_timesteps, b).to(vid.device)
        # time = self.scheduler(torch.rand(b)/b + torch.arange(0, b)/b).to(img.device)
        eps = torch.randn_like(vid)
        n_img = self.sampler.add_noise(vid, eps, time)

        pred = self.model(n_img, time, txt, mask, temporal_mask=self.temporal_mask)

        if self.mode == "fm":
            target = (eps - vid)
            loss = {"loss": ((target - pred)**2).mean()}
        elif self.mode == "eps":
            loss = {"loss": ((pred - eps)**2).mean()}

        # logging
        for metric_name, metric_value in loss.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if hasattr(self, "do_optimizer_step") and not self.do_optimizer_step:
            print("Skipping optimizer step")
            closure_result = optimizer_closure()
            if closure_result is not None:
                return closure_result
            else:
                return
        else:
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def configure_optimizers(self):
        if self.optimizer_cfg.exclude_ln_and_biases_from_weight_decay:
            print("Removing LN, Embedding and biases from weight decay")
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm, nn.Embedding])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.optimizer_cfg.optim.keywords["weight_decay"],
                    "layer_adaptation": True,  # for lamb
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,  # for lamb
                },
            ]
            optimizer = self.optimizer_cfg.optim(optimizer_grouped_parameters)
        else:
            optimizer = self.optimizer_cfg.optim(self.model.parameters())
        scheduler = self.lr_scheduler_builder(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
