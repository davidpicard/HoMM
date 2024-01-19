import lightning.pytorch as pl
import torch

from gluonts.core.component import validated
from gluonts.itertools import select

from .module import PatchHoMModel


class PatchHoMLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``PatchHoMModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``PatchHoMModel`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to construct the ``PatchHoMModel`` to be trained.
    loss
        Loss function to be used for training.
    lr
        Learning rate.
    weight_decay
        Weight decay regularization parameter.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = PatchHoMModel(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.inputs = self.model.describe_inputs()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self.model.loss(
            **select(self.inputs, batch),
            future_target=batch["future_target"],
            future_observed_values=batch["future_observed_values"],
        ).mean()

        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self.model.loss(
            **select(self.inputs, batch),
            future_target=batch["future_target"],
            future_observed_values=batch["future_observed_values"],
        ).mean()

        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
