import os
from typing import Any
import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
import copy
import numpy as np


class ClassificationModule(L.LightningModule):
    def __init__(
        self,
        model,
        loss,
        optimizer_cfg,
        lr_scheduler_builder,
        train_batch_preprocess,
        train_metrics,
        val_metrics,
        test_metrics,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_builder = lr_scheduler_builder
        self.train_batch_preprocess = train_batch_preprocess
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def training_step(self, batch, batch_idx):
        img, label = batch
        img, label = self.train_batch_preprocess(img, label)
        pred = self.model(img)
        loss = self.loss(pred, label, average=True)
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        self.train_metrics.update(pred, label)
        return loss

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.model(img)
        loss = self.loss(pred, label, average=True)
        self.val_metrics(pred, label)
        self.log("val/loss", loss["loss"], sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):
        img, label = batch
        pred = self.model(img)
        loss = self.loss(pred, label, average=True)
        self.test_metrics.update(pred, label)
        self.log(
            "test/loss", loss["loss"], sync_dist=True, on_step=False, on_epoch=True
        )

    def test_epoch_end(self, outputs):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        if self.optimizer_cfg.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
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
                    "weight_decay": self.optimizer_cfg.optimizer.optim.weight_decay,
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
