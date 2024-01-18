from typing import Optional, Iterable, Dict, Any

import torch
import lightning.pytorch as pl

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.transform import (
    Transformation,
    AddObservedValuesIndicator,
    InstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import Output, StudentTOutput

from .lightning_module import PatchHoMLightningModule

PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values"]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class PatchHoMEstimator(PyTorchLightningEstimator):
    """
    An estimator training the PatchHoM model for forecasting.

    This class uses the model defined in ``PatchHoMModel``,
    and wraps it into a ``PatchHoMLightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``10 * prediction_length``).
    patch_len
        Length of the patch.
    stride
        Stride of the patch.
    padding_patch
        Padding of the patch.
    d_model
        Size of hidden layers in the HoM encoder.
    order
        Order of the moments.
    order_expand:
        TODO
    ffw_expand
        Size of hidden layers in the HoM encoder.
    dropout
        Dropout probability in the HoM encoder.
    num_encoder_layers
        Number of the HoM encoder.
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).
    scaling
        Scaling parameter can be "mean", "std" or None.
    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch
            (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        patch_len: int,
        context_length: Optional[int] = None,
        stride: int = 8,
        padding_patch: str = "end",
        d_model: int = 32,
        order: int = 4,
        order_expand: int = 8,
        ffw_expand: int = 4,
        dropout: float = 0.1,
        num_encoder_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        scaling: Optional[str] = "mean",
        distr_output: Output = StudentTOutput(),
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.prediction_length = prediction_length
        self.context_length = context_length or 10 * prediction_length
        # TODO find way to enforce same defaults to network and estimator
        # somehow
        self.lr = lr
        self.weight_decay = weight_decay
        self.distr_output = distr_output
        self.scaling = scaling
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.d_model = d_model
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        return SelectFields(
            [
                FieldName.ITEM_ID,
                FieldName.INFO,
                FieldName.START,
                FieldName.TARGET,
            ],
            allow_missing=True,
        ) + AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

    def create_lightning_module(self) -> pl.LightningModule:
        return PatchHoMLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            model_kwargs={
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,
                "patch_len": self.patch_len,
                "stride": self.stride,
                "padding_patch": self.padding_patch,
                "d_model": self.d_model,
                "order": self.order,
                "order_expand": self.order_expand,
                "ffw_expand": self.ffw_expand,
                "dropout": self.dropout,
                "num_encoder_layers": self.num_encoder_layers,
                "distr_output": self.distr_output,
                "scaling": self.scaling,
            },
        )

    def _create_instance_splitter(self, module: PatchHoMLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: PatchHoMLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self, data: Dataset, module: PatchHoMLightningModule, **kwargs
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_predictor(
        self, transformation: Transformation, module
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            forecast_generator=self.distr_output.forecast_generator,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )
