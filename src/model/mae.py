import os
from typing import Any
import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
import copy
import numpy as np
from torchvision import transforms
import einops


class MAEModule(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        mask_prob,
        loss,
        optimizer_cfg,
        lr_scheduler_builder,
        train_batch_preprocess,
        train_metrics,
        val_metrics,
        test_metrics,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.mask_prob = mask_prob
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_builder = lr_scheduler_builder
        self.train_batch_preprocess = train_batch_preprocess
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def _compose_masks(self, batch):
        imgs, _ = batch
        b, c, h, w = imgs.shape
        
        n_tokens = (self.encoder.im_size//self.encoder.kernel_size)**2
        mask = torch.bernoulli(torch.empty((b, n_tokens)).uniform_(0, 1), p=self.mask_prob).to(self.device)
        masked_imgs = einops.rearrange(imgs, 'b c (m h) (n w) -> b (m n) (h w c)', c=3,
                                       m=self.encoder.im_size // self.encoder.kernel_size, w=self.encoder.kernel_size)
        masked_imgs = masked_imgs * mask.unsqueeze(-1)
        masked_imgs = einops.rearrange(masked_imgs, 'b (m n) (h w c) -> b c (m h) (n w)', c=3,
                                       m=self.encoder.im_size // self.encoder.kernel_size, w=self.encoder.kernel_size)
        return masked_imgs, mask

    def training_step(self, batch, batch_idx):
        masked_imgs, mask = self._compose_masks(batch)
        imgs, _ = batch
        b, c, h, w = imgs.shape

        imgs_enc = self.encoder(masked_imgs, mask)  # attend only unmask tokens using mask
        outputs = self.decoder(
            imgs_enc * mask.unsqueeze(-1))  # attend all tokens but don't backprop masked ones to the encoder
        outputs = einops.rearrange(outputs, 'b (m n) (h w c) -> b c (m h) (n w)', c=3,
                                   m=self.encoder.im_size // self.encoder.kernel_size, w=self.encoder.kernel_size)

        loss = self.loss(outputs, imgs)

        if batch_idx % 50 == 0:
            denormalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
            )
            origin = denormalize(imgs[0]).squeeze(0)
            masked = denormalize(masked_imgs[0]).squeeze(0)
            recons = denormalize(outputs[0].float()).squeeze(0)
            self.logger.log_image(key="samples", images=[origin, masked, recons], caption=["origin", "masked", "recons"])

        self.log(f"train/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        masked_imgs, mask = self._compose_masks(batch)
        imgs, _ = batch
        b, c, h, w = imgs.shape
         
        imgs_enc = self.encoder(masked_imgs, mask)  # attend only unmask tokens using mask
        outputs = self.decoder(
            imgs_enc * mask.unsqueeze(-1))  # attend all tokens but don't backprop masked ones to the encoder
        outputs = einops.rearrange(outputs, 'b (m n) (h w c) -> b c (m h) (n w)', c=3,
                                   m=self.encoder.im_size // self.encoder.kernel_size, w=self.encoder.kernel_size)
        loss = self.loss(outputs, imgs)
        self.log("val/loss", loss.item(), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        masked_imgs, mask = self._compose_masks(batch)
        imgs, _ = batch
        b, c, h, w = imgs.shape

        imgs_enc = self.encoder(masked_imgs, mask)  # attend only unmask tokens using mask
        outputs = self.decoder(
            imgs_enc * mask.unsqueeze(-1))  # attend all tokens but don't backprop masked ones to the encoder
        outputs = einops.rearrange(outputs, 'b (m n) (h w c) -> b c (m h) (n w)', c=3,
                                   m=h // self.encoder.kernel_size, w=self.encoder.kernel_size)

        loss = self.loss(outputs, imgs)
        self.log("test/loss", loss.item(), on_step=False, on_epoch=True)

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        if self.optimizer_cfg.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.encoder, [nn.LayerNorm]) \
                                  + get_parameter_names(self.decoder, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.encoder.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.optimizer_cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,  # for lamb
                },
                {
                    "params": [
                        p
                        for n, p in self.encoder.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,  # for lamb
                },
                {
                    "params": [
                        p
                        for n, p in self.decoder.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.optimizer_cfg.optimizer.optim.weight_decay,
                    "layer_adaptation": True,  # for lamb
                },
                {
                    "params": [
                        p
                        for n, p in self.decoder.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                    "layer_adaptation": False,  # for lamb
                },
            ]
            optimizer = self.optimizer_cfg.optim(optimizer_grouped_parameters)
        else:
            optimizer = self.optimizer_cfg.optim(list(self.encoder.parameters())+list(self.decoder.parameters()))
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
